# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import asyncio
import json
import logging
import sys
import textwrap
import uuid
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Dict, List, NamedTuple
from urllib.parse import urlparse

import fsspec
from anyio import Path
from comfy_catapult.comfy_schema import APIObjectInfo, APIWorkflow
from datauri import DataURI
from fastapi import FastAPI
from pydantic import ValidationError
from rich.console import Console

from . import _build_version
from .args import (Arguments, ClientCommand, ClientExecuteCommand,
                   ClientUploadCommand, ExecuteArgs, ExecuteCommand, FSArg,
                   Parse, ProvisionerArgs, ServeCommand, UploadArgs)
from .comfyfs import RegisterComfyUIFS
from .error_utils import _ErrorContext, _HTTPError, _SuccessOrExit, _YamlDump
from .log import _SetupLogging
from .lowda import (DumbProvisioner, FSSpecRemoteFileAPI, ProvisionerBase,
                    Server, ServerBase, _ParseUserIOSpec)
from .lowda_types import (ComfyRemoteInfo, InputMapping, InputPPKind, IOSpec,
                          JSONSerializable, OutputMapping, OutputPPKind,
                          ProvisioningSpec, UserIOSpec, WorkflowTemplateBundle)
from .remote_lowda import RemoteServer
from .validation import _CheckModelRoundTrip, _DoubleValidateJSON
from .web.app import InitializeFastAPI, RunFastAPI
from .web.settings import AppSettings

logger = logging.getLogger(__name__)


async def GetProvisioner(cmd: ProvisionerArgs) -> ProvisionerBase:
  comfy_api_url = cmd.comfy_api_url
  if not comfy_api_url.endswith('/'):
    comfy_api_url += '/'
  provisioner = DumbProvisioner(comfy_remote=ComfyRemoteInfo(
      comfy_api_url=comfy_api_url,
      upload={},
      download={
          'temp/': IOSpec(io_url=f'comfy+{comfy_api_url}temp/'),
          'output/': IOSpec(io_url=f'comfy+{comfy_api_url}output/'),
      },
      logs=None))
  return provisioner


async def GetFSSpecRemoteFileAPI(fs_args: List[FSArg],
                                 console: Console) -> FSSpecRemoteFileAPI:
  remote = FSSpecRemoteFileAPI(overwrite=FSSpecRemoteFileAPI.Overwrite.RENAME)
  remote.AddFS(url_prefix='comfy+http://',
               fs=fsspec.filesystem('comfy+http'),
               mode='rw')
  remote.AddFS(url_prefix='comfy+https://',
               fs=fsspec.filesystem('comfy+https'),
               mode='rw')
  fs_arg: FSArg
  for fs_arg in fs_args:
    remote.AddFS(url_prefix=fs_arg.prefix,
                 fs=fsspec.filesystem(fs_arg.protocol),
                 mode=fs_arg.mode)
  return remote


async def Serve(args: Arguments, cmd: ServeCommand, console: Console) -> int:
  await _SetupLogging(cmd)
  provisioner = await GetProvisioner(cmd)
  remote = await GetFSSpecRemoteFileAPI(cmd.fs, console)
  server = await Server.Create(provisioner=provisioner,
                               remote=remote,
                               tmp_dir_path=await
                               Path(cmd.tmp_dir_path).absolute(),
                               debug_path=await Path(cmd.debug_path).absolute(),
                               debug_save_all=cmd.debug_save_all)
  settings = AppSettings(
      project_name='Comfylowda',
      version=_build_version,
      description='Comfy Catapult Lowda',
      docs_url='/docs',
      redoc_url='/redoc',
      debug=cmd.debug,
      hostname=cmd.bind,
      port=cmd.port,
      allowed_hosts=['*'],
  )
  app: FastAPI = await InitializeFastAPI(settings=settings)
  app.state.server = server
  await RunFastAPI(app=app, settings=settings)
  return 0


async def _ConstructUploadReq(
    upload_args: UploadArgs, workflow_id: str, console: Console,
    error_context: _ErrorContext) -> ServerBase.UploadWorkflowReq:
  error_context['upload_args'] = await error_context.LargeToFile(
      'upload_args',
      upload_args.model_dump(mode='json', round_trip=True, by_alias=True))

  try:
    api_workflow_template_content = await Path(upload_args.api_workflow
                                               ).read_text()
    api_workflow_template = _DoubleValidateJSON(
        arg_name='--api-workflow-template',
        json_str=api_workflow_template_content,
        t=APIWorkflow)
    object_info_content = await Path(upload_args.object_info).read_text()
    object_info = _DoubleValidateJSON(arg_name='--object-info',
                                      json_str=object_info_content,
                                      t=APIObjectInfo)

    input_mappings: List[InputMapping] = []
    im_json_str: str
    for im_json_str in upload_args.im:
      im = _DoubleValidateJSON(arg_name='-im',
                               json_str=im_json_str,
                               t=InputMapping)
      input_mappings.append(im)

    output_mappings: List[OutputMapping] = []
    om_json_str: str
    for om_json_str in upload_args.om:
      om = _DoubleValidateJSON(arg_name='-om',
                               json_str=om_json_str,
                               t=OutputMapping)
      output_mappings.append(om)
  except ValidationError:
    console.print_exception()
    console.print('Failed to validate inputs. Exiting.', style='red')
    sys.exit(1)
  ############################################################################

  template_bundle: WorkflowTemplateBundle
  template_bundle: WorkflowTemplateBundle = WorkflowTemplateBundle(
      api_workflow_template=api_workflow_template,
      important=[],
      object_info=object_info,
      input_mappings=input_mappings,
      output_mappings=output_mappings)
  # workflow_id = sha256(
  #     template_bundle.model_dump_json(
  #         round_trip=True, by_alias=True).encode('utf-8')).hexdigest()
  # TODO: Put this on the command line.
  prov_spec = ProvisioningSpec(files={}, archives={}, custom_nodes={})
  req = ServerBase.UploadWorkflowReq(workflow_id=workflow_id,
                                     template_bundle=template_bundle,
                                     prov_spec=prov_spec)
  error_context['req'] = await error_context.LargeToFile(
      'req', req.model_dump(mode='json', round_trip=True, by_alias=True))

  try:
    _CheckModelRoundTrip(model=req, t=ServerBase.UploadWorkflowReq)
  except Exception:
    console.print_exception()
    console.print('Failed to validate final request. Exiting.', style='red')
    sys.exit(1)
  return req


async def ClientUploadWorkflow(args: Arguments, client_cmd: ClientCommand,
                               cmd: ClientUploadCommand,
                               console: Console) -> int:
  debug_path: Path = await Path(client_cmd.debug_path).absolute()
  error_context = await _ErrorContext.Create(debug_path=debug_path,
                                             key='ClientUpload')
  try:
    error_context['cmd'] = await error_context.LargeToFile(
        'cmd', cmd.model_dump(mode='json', round_trip=True, by_alias=True))

    ############################################################################
    req = await _ConstructUploadReq(upload_args=cmd,
                                    workflow_id=cmd.workflow_id,
                                    console=console,
                                    error_context=error_context.Copy())
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True, by_alias=True))

    logger.info('Uploading workflow...',
                extra={'error_context': error_context.Dump()})

    server = RemoteServer(lowda_url=client_cmd.lowda_url,
                          debug_path=debug_path,
                          console=console)
    success: ServerBase.UploadWorkflowSuccess
    success = await _SuccessOrExit(action='upload workflow',
                                   res=await server.UploadWorkflow(req),
                                   console=console)
    error_context['res.success'] = success.model_dump(mode='json',
                                                      round_trip=True,
                                                      by_alias=True)
    logger.info('Uploaded workflow successfully.',
                extra={'error_context': error_context.Dump()})
    return 0
  except _HTTPError as e:
    console.print('HTTP Error while uploading workflow.', style='red')
    console.print('HTTP status:', e.status, style='red')
    console.print('HTTP reason:', e.reason, style='red')
    if e.response is not None:
      console.print('HTTP response:\n' + textwrap.indent(e.response, '  '),
                    style='red')
    else:
      console.print('HTTP response:', e.response, style='red')
    console.print('Failed to upload workflow. Exiting.', style='red')
    return 1
  except Exception:
    logger.exception('Failed to upload workflow.',
                     extra={'error_context': error_context.Dump()})
    console.print_exception()
    console.print('Failed to upload workflow. Exiting.', style='red')
    return 1


class LocalFileTranslatorBase(ABC):

  class PostProcessInfo(NamedTuple):
    original_user_input_values: Dict[str, JSONSerializable]

  @abstractmethod
  async def Pretranslate(
      self, *, server: ServerBase, req: ServerBase.ExecuteReq,
      template_bundle: WorkflowTemplateBundle) -> PostProcessInfo:
    raise NotImplementedError()

  @abstractmethod
  async def PostTranslate(self, *, server: ServerBase,
                          req: ServerBase.ExecuteReq,
                          res: ServerBase.ExecuteSuccess,
                          template_bundle: WorkflowTemplateBundle,
                          post_translate_info: PostProcessInfo):
    raise NotImplementedError()


class B64LocalFileTranslator(LocalFileTranslatorBase):
  PostTranslateInfo = LocalFileTranslatorBase.PostProcessInfo

  async def _PretranslateValue(self, *, server: ServerBase,
                               req: ServerBase.ExecuteReq,
                               template_bundle: WorkflowTemplateBundle,
                               user_input_name: str):
    input_mappings: Dict[str, InputMapping] = dict([
        (im.name, im) for im in template_bundle.input_mappings
    ])
    output_mappings: Dict[str, OutputMapping] = dict([
        (om.name, om) for om in template_bundle.output_mappings
    ])
    if user_input_name in input_mappings:
      input_mapping: InputMapping = input_mappings[user_input_name]
      if input_mapping.pp != InputPPKind.FILE:
        return
      user_input_value = req.user_input_values[user_input_name]
      src_io_spec = _ParseUserIOSpec(user_input_value).ToIOSpec()
      if not src_io_spec.io_url.startswith('file:///'):
        return
      # Turn file uri into a local path
      src_path = Path(urlparse(src_io_spec.io_url).path)
      if not src_path.exists():
        raise FileNotFoundError(f'File not found: {json.dumps(str(src_path))}')
      if not src_path.is_file():
        raise FileNotFoundError(f'Not a file: {json.dumps(str(src_path))}')

      async with await src_path.open('rb') as f:
        data: bytes = await f.read()
        data_uri = DataURI.make(data=data,
                                base64=True,
                                charset=None,
                                mimetype=None)
        req.user_input_values[user_input_name] = UserIOSpec(str(data_uri))
    elif user_input_name in output_mappings:
      output_mapping: OutputMapping = output_mappings[user_input_name]
      if output_mapping.pp != OutputPPKind.FILE:
        return
      user_input_value = req.user_input_values[user_input_name]
      dest_io_spec = _ParseUserIOSpec(user_input_value).ToIOSpec()
      if not dest_io_spec.io_url.startswith('file:///'):
        return
      req.user_input_values[user_input_name] = UserIOSpec('base64').model_dump(
          mode='json', round_trip=True, by_alias=True)

  async def Pretranslate(
      self, *, server: ServerBase, req: ServerBase.ExecuteReq,
      template_bundle: WorkflowTemplateBundle) -> PostTranslateInfo:
    tasks: List[asyncio.Task] = []
    # Shallow copy.
    original_user_input_values = req.user_input_values.copy()
    for name in req.user_input_values.keys():
      tasks.append(
          asyncio.create_task(
              self._PretranslateValue(server=server,
                                      req=req,
                                      template_bundle=template_bundle,
                                      user_input_name=name)))
    await asyncio.gather(*tasks)
    return LocalFileTranslatorBase.PostProcessInfo(
        original_user_input_values=original_user_input_values)

  async def _PostTranslateValue(self, *, server: ServerBase,
                                req: ServerBase.ExecuteReq,
                                res: ServerBase.ExecuteSuccess,
                                template_bundle: WorkflowTemplateBundle,
                                user_input_name: str,
                                post_translate_info: PostTranslateInfo):
    output_mappings: Dict[str, OutputMapping] = dict([
        (om.name, om) for om in template_bundle.output_mappings
    ])
    if user_input_name not in output_mappings:
      return
    output_mapping: OutputMapping = output_mappings[user_input_name]
    if output_mapping.pp != OutputPPKind.FILE:
      return
    user_input_value0 = post_translate_info.original_user_input_values[
        user_input_name]

    dest_io_spec0 = _ParseUserIOSpec(user_input_value0).ToIOSpec()
    if not dest_io_spec0.io_url.startswith('file:///'):
      return

    user_input_value1 = req.user_input_values[user_input_name]
    dest_io_spec1 = _ParseUserIOSpec(user_input_value1).ToIOSpec()

    if not dest_io_spec1.io_url.startswith('base64'):
      raise ValueError(
          f'Expected base64 data for output file {json.dumps(user_input_name)}.'
      )
    if user_input_name not in res.mapped_outputs:
      raise ValueError(
          f'Output file {json.dumps(user_input_name)} not found in output.')

    output_value = res.mapped_outputs[user_input_name]
    data_uri = DataURI(output_value)
    data = data_uri.data
    dest_path = Path(urlparse(dest_io_spec0.io_url).path)
    async with await dest_path.open('wb') as f:
      await f.write(data)
    res.mapped_outputs[user_input_name] = dest_io_spec0.io_url

  async def PostTranslate(self, *, server: ServerBase,
                          req: ServerBase.ExecuteReq,
                          res: ServerBase.ExecuteSuccess,
                          template_bundle: WorkflowTemplateBundle,
                          post_translate_info: PostTranslateInfo):
    tasks: List[asyncio.Task] = []
    for name in post_translate_info.original_user_input_values.keys():
      tasks.append(
          asyncio.create_task(
              self._PostTranslateValue(
                  server=server,
                  req=req,
                  res=res,
                  template_bundle=template_bundle,
                  user_input_name=name,
                  post_translate_info=post_translate_info)))
    await asyncio.gather(*tasks)


async def _ConstructExecuteReq(
    execute_args: ExecuteArgs, workflow_id: str, console: Console,
    error_context: _ErrorContext) -> ServerBase.ExecuteReq:
  error_context['execute_args'] = await error_context.LargeToFile(
      'execute_args',
      execute_args.model_dump(mode='json', round_trip=True, by_alias=True))

  user_input_values: Dict[str, JSONSerializable] = {}
  try:
    for i in execute_args.i:
      try:
        name: str = i.name
        value = json.loads(i.json_value)
        user_input_values[name] = value
      except JSONDecodeError as e:
        raise ValueError(
            f'Failed to parse JSON input for argument -i with name={json.dumps(name)}.'
        ) from e
    keepalive: float = execute_args.keepalive
  except Exception:
    console.print_exception()
    console.print('Failed to validate inputs. Exiting.', style='red')
    sys.exit(1)
  ############################################################################
  job_id = uuid.uuid4().hex
  ############################################################################

  req = ServerBase.ExecuteReq(job_id=job_id,
                              workflow_id=workflow_id,
                              user_input_values=user_input_values,
                              keepalive=keepalive)
  error_context['req'] = await error_context.LargeToFile(
      'req', req.model_dump(mode='json', round_trip=True, by_alias=True))

  try:
    _CheckModelRoundTrip(model=req, t=ServerBase.ExecuteReq)
  except Exception:
    console.print_exception()
    console.print('Failed to validate final request. Exiting.', style='red')
    sys.exit(1)
  return req


async def ClientExecute(args: Arguments, client_cmd: ClientCommand,
                        cmd: ClientExecuteCommand, console: Console) -> int:

  debug_path: Path = await Path(client_cmd.debug_path).absolute()
  error_context = await _ErrorContext.Create(debug_path=debug_path,
                                             key='ClientExecute')
  try:
    error_context['cmd'] = await error_context.LargeToFile(
        'cmd', cmd.model_dump(mode='json', round_trip=True, by_alias=True))

    server = RemoteServer(lowda_url=client_cmd.lowda_url,
                          debug_path=debug_path,
                          console=console)
    ############################################################################
    # File inputs of the schema file:/// will not be available on the Lowda
    # Server, so we need to transfer them to the server.
    download_success = await _SuccessOrExit(
        action='download workflow',
        res=await server.DownloadWorkflow(
            ServerBase.DownloadWorkflowReq(workflow_id=cmd.workflow_id)),
        console=console)
    template_bundle: WorkflowTemplateBundle = download_success.template_bundle
    ############################################################################
    req: ServerBase.ExecuteReq
    req = await _ConstructExecuteReq(execute_args=cmd,
                                     console=console,
                                     workflow_id=cmd.workflow_id,
                                     error_context=error_context.Copy())

    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True, by_alias=True))

    local_file_translator: LocalFileTranslatorBase = B64LocalFileTranslator()
    pp_info = await local_file_translator.Pretranslate(
        server=server, req=req, template_bundle=template_bundle)
    error_context['req_pre_translated'] = await error_context.LargeToFile(
        'req_pre_translated',
        req.model_dump(mode='json', round_trip=True, by_alias=True))

    logger.info('Executing remote workflow...',
                extra={'error_context': error_context.Dump()})

    res: ServerBase.ExecuteRes
    res = await server.Execute(req)
    error_context['res'] = res.model_dump(mode='json',
                                          round_trip=True,
                                          by_alias=True)
    success: ServerBase.ExecuteSuccess
    success = await _SuccessOrExit(action='execute remote workflow',
                                   res=res,
                                   console=console)

    logger.info('Executed remote workflow successfully.',
                extra={'error_context': error_context.Dump()})

    await local_file_translator.PostTranslate(server=server,
                                              req=req,
                                              res=success,
                                              template_bundle=template_bundle,
                                              post_translate_info=pp_info)
    print(json.dumps(success.mapped_outputs))
    return 0
  except _HTTPError as e:
    console.print('HTTP Error while executing remote workflow.', style='red')
    console.print('HTTP status:', e.status, style='red')
    console.print('HTTP reason:', e.reason, style='red')
    if e.response is not None:
      console.print('HTTP response:\n' + textwrap.indent(e.response, '  '),
                    style='red')
    else:
      console.print('HTTP response:', e.response, style='red')
    console.print('Failed to execute remote workflow. Exiting.', style='red')
    return 1
  except Exception:
    logger.exception('Failed to execute remote workflow.',
                     extra={'error_context': error_context.Dump()})
    console.print_exception()
    console.print('Failed to execute remote workflow. Exiting.', style='red')
    return 1


async def Client(args: Arguments, cmd: ClientCommand, console: Console) -> int:
  await _SetupLogging(cmd)
  if cmd.upload is not None:
    return await ClientUploadWorkflow(args, cmd, cmd.upload, console)
  elif cmd.execute is not None:
    return await ClientExecute(args, cmd, cmd.execute, console)
  else:
    console.print('No command specified. Exiting.', style='red')
    sys.exit(1)


async def Execute(args: Arguments, cmd: ExecuteCommand,
                  console: Console) -> int:
  await _SetupLogging(cmd)

  error_context = await _ErrorContext.Create(debug_path=await
                                             Path(cmd.debug_path).absolute(),
                                             key='Execute')
  provisioner = await GetProvisioner(cmd)
  remote = await GetFSSpecRemoteFileAPI(cmd.fs, console)
  server = await Server.Create(provisioner=provisioner,
                               remote=remote,
                               tmp_dir_path=await
                               Path(cmd.tmp_dir_path).absolute(),
                               debug_path=await Path(cmd.debug_path).absolute(),
                               debug_save_all=cmd.debug_save_all)
  workflow_id = uuid.uuid4().hex
  upload_req = await _ConstructUploadReq(upload_args=cmd,
                                         workflow_id=workflow_id,
                                         console=console,
                                         error_context=error_context.Copy())
  execute_req = await _ConstructExecuteReq(execute_args=cmd,
                                           workflow_id=workflow_id,
                                           console=console,
                                           error_context=error_context.Copy())
  local_file_translator: LocalFileTranslatorBase = B64LocalFileTranslator()
  pp_info = await local_file_translator.Pretranslate(
      server=server,
      req=execute_req,
      template_bundle=upload_req.template_bundle)

  _ = await _SuccessOrExit(action='upload workflow',
                           res=await server.UploadWorkflow(upload_req),
                           console=console)

  execute_success: ServerBase.ExecuteSuccess
  execute_success = await _SuccessOrExit(action='execute workflow',
                                         res=await server.Execute(execute_req),
                                         console=console)
  await local_file_translator.PostTranslate(
      server=server,
      req=execute_req,
      res=execute_success,
      template_bundle=upload_req.template_bundle,
      post_translate_info=pp_info)
  console.print('Workflow executed successfully.', style='green')
  console.print('Success:',
                _YamlDump(
                    execute_success.model_dump(mode='json',
                                               round_trip=True,
                                               by_alias=True)),
                style='green')
  console.print('Workflow executed successfully.', style='green')
  print(json.dumps(execute_success.mapped_outputs))
  return 0


async def amain() -> int:

  console = Console(file=sys.stderr)
  try:
    RegisterComfyUIFS()
    ############################################################################
    args: Arguments = Parse(version=_build_version)
    ############################################################################
    if args.serve is not None:
      return await Serve(args, args.serve, console)
    elif args.client is not None:
      return await Client(args, args.client, console)
    elif args.execute is not None:
      return await Execute(args, args.execute, console)
    else:
      console.print('No command specified. Exiting.', style='red')
      sys.exit(1)
    ############################################################################
  except Exception:
    console.print_exception()
    console.print('Failed to execute the workflow. Exiting.', style='red')
    sys.exit(1)


if __name__ == '__main__':
  sys.exit(asyncio.run(amain()))
