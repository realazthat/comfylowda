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
from json import JSONDecodeError
from typing import Dict, List, Tuple

import aiohttp
import fsspec
import httpx
import requests
from anyio import Path
from comfy_catapult.comfy_schema import APIObjectInfo, APIWorkflow
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from rich.console import Console
from typing_extensions import TypeVar

from . import _build_version
from .args import (Arguments, ClientCommand, ClientExecuteCommand,
                   ClientUploadCommand, ExecuteArgs, ExecuteCommand, FSArg,
                   Parse, ProvisionerArgs, ServeCommand, UploadArgs)
from .comfy_schema import Workflow
from .comfyfs import RegisterComfyUIFS
from .error_utils import _ErrorContext, _YamlDump
from .log import _SetupLogging
from .lowda import (DumbProvisioner, FSSpecRemoteFileAPI, ProvisionerBase,
                    Server, ServerBase)
from .lowda_types import (ComfyRemoteInfo, InputMapping, IOSpec,
                          JSONSerializable, OutputMapping, ProvisioningSpec,
                          WorkflowTemplateBundle)
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


class _HTTPError(Exception):

  def __init__(self, status: int, reason: str | None,
               response: str | None) -> None:
    self.status: int = status
    self.reason: str | None = reason
    self.response: str | None = response


_RequestModelT = TypeVar('_RequestModelT', bound=BaseModel)
_ResponseModelT = TypeVar('_ResponseModelT', bound=BaseModel)


async def _ClientUploadRequestsPost(action: str, url: str, req: BaseModel,
                                    resp_type: type[_ResponseModelT],
                                    error_context: _ErrorContext,
                                    console: Console) -> _ResponseModelT:
  response: requests.Response
  response = await asyncio.to_thread(requests.post,
                                     url,
                                     json=req.model_dump(mode='json',
                                                         round_trip=True,
                                                         by_alias=True),
                                     timeout=100)

  error_context['response.headers'] = dict(response.headers)
  error_context['response.status_code'] = response.status_code
  error_context['response.reason'] = response.reason
  error_context['response.ok'] = response.ok
  if not response.ok:
    response_text: str | None = None
    try:
      response_json_dict = await asyncio.to_thread(response.json)
      error_context['response_json_dict'] = response_json_dict
      response_text = _YamlDump(response_json_dict)
      error_context['response_json_text_yaml'] = response_text
    except JSONDecodeError as e:
      error_context['response_json_dict_error'] = str(e)
    if response_text is None:
      response_text = response.text
    error_context['response_text'] = response_text

    console.print(f'Failed to {action}.', style='red')
    console.print('Headers:\n' +
                  textwrap.indent(_YamlDump(dict(response.headers)), '  '),
                  style='red')
    console.print('Error context:\n' +
                  textwrap.indent(_YamlDump(error_context.Dump()), '  '),
                  style='red')
    console.print(f'Failed to {action}. Exiting.', style='red')
    raise _HTTPError(status=response.status_code,
                     reason=response.reason,
                     response=response_text)
  response_dict = await asyncio.to_thread(response.json)
  return resp_type.model_validate(response_dict)


async def _ClientUploadHTTPXPost(action: str, url: str,
                                 req: ServerBase.UploadWorkflowReq,
                                 resp_type: type[_ResponseModelT],
                                 error_context: _ErrorContext,
                                 console: Console) -> _ResponseModelT:
  async with httpx.AsyncClient() as client:
    response = await client.post(url,
                                 json=req.model_dump(mode='json',
                                                     round_trip=True,
                                                     by_alias=True))

    error_context['response.http_version'] = response.http_version
    error_context['response.headers'] = response.headers.multi_items()
    error_context['response.status_code'] = response.status_code
    error_context['response.reason_phrase'] = response.reason_phrase
    error_context['response.is_success'] = response.is_success
    error_context['response.is_error'] = response.is_closed
    error_context['response.is_error'] = response.is_error
    error_context['response.is_server_error'] = response.is_server_error
    error_context['response.is_client_error'] = response.is_client_error
    if not response.is_success:
      try:
        response_dict = await response.json()
        error_context['response_dict'] = response_dict
      except JSONDecodeError as e:
        error_context['response_dict_error'] = str(e)
      console.print(f'Failed to {action}.', style='red')
      console.print(
          'Headers:\n' +
          textwrap.indent(_YamlDump(response.headers.multi_items()), '  '),
          style='red')
      console.print('Error context:\n' +
                    textwrap.indent(_YamlDump(error_context.Dump()), '  '),
                    style='red')
      console.print(f'Failed to {action}. Exiting.', style='red')
      raise _HTTPError(status=response.status_code,
                       reason=response.reason_phrase,
                       response=response.text)
    response_dict = await response.json()
    return resp_type.model_validate(response_dict)


async def _ClientUploadAIOHTTPPost(action: str, url: str,
                                   req: ServerBase.UploadWorkflowReq,
                                   resp_type: type[_ResponseModelT],
                                   error_context: _ErrorContext,
                                   console: Console) -> _ResponseModelT:
  async with aiohttp.ClientSession() as client:

    async with await client.post(url,
                                 json=req.model_dump(
                                     mode='json',
                                     round_trip=True,
                                     by_alias=True)) as response:
      headers_list: List[Tuple[str, str]] = []
      for k, v in response.headers.items():
        headers_list.append((str(k), v))
      error_context['response.version'] = (tuple(
          response.version) if response.version is not None else None)
      error_context['response.headers'] = headers_list
      error_context['response.status_code'] = response.status
      error_context['response.reason_phrase'] = response.reason
      error_context['response.ok'] = response.ok
      if not response.ok:
        response_text: str | None = None
        try:
          response_dict = await response.json()
          error_context['response_dict'] = response_dict
          response_text = _YamlDump(response_dict)
          error_context['response_dict_yaml'] = response_text
        except Exception as e:
          error_context['response_dict_error'] = str(e)

        if response_text is None:
          try:
            response_text = await response.text()
            error_context['response_text'] = response_text
          except Exception as e:
            error_context['response_text_error'] = str(e)
        console.print(f'Failed to {action}.', style='red')
        console.print('Headers:\n' +
                      textwrap.indent(_YamlDump(headers_list), '  '),
                      style='red')
        console.print('Error context:\n' +
                      textwrap.indent(_YamlDump(error_context.Dump()), '  '),
                      style='red')
        console.print(f'Failed to {action}. Exiting.', style='red')
        raise _HTTPError(status=response.status,
                         reason=response.reason,
                         response=response_text)
      response_dict = await response.json()
      return resp_type.model_validate(response_dict)


async def _ConstructUploadReq(
    upload_args: UploadArgs, workflow_id: str, console: Console,
    error_context: _ErrorContext) -> ServerBase.UploadWorkflowReq:
  error_context['upload_args'] = await error_context.LargeToFile(
      'upload_args',
      upload_args.model_dump(mode='json', round_trip=True, by_alias=True))

  try:
    workflow_template_content = await Path(upload_args.workflow).read_text()
    workflow_template = _DoubleValidateJSON(arg_name='--workflow-template',
                                            json_str=workflow_template_content,
                                            t=Workflow)
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
      workflow_template=workflow_template,
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

    url = client_cmd.lowda_url
    if url.endswith('/'):
      url = url[:-1]
    url += '/upload/'
    error_context['url'] = url
    logger.info('Uploading workflow...',
                extra={'error_context': error_context.Dump()})

    res: ServerBase.UploadWorkflowRes
    # res = await _ClientUploadAIOHTTPPost(url, req, error_context.Copy(), console)
    # res = await _ClientUploadHTTPXPost(url, req, error_context.Copy(), console)
    res = await _ClientUploadRequestsPost(
        action='upload workflow',
        url=url,
        req=req,
        resp_type=ServerBase.UploadWorkflowRes,
        error_context=error_context.Copy(),
        console=console)
    error_context['res'] = res.model_dump(mode='json',
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

    ############################################################################
    req: ServerBase.ExecuteReq
    req = await _ConstructExecuteReq(execute_args=cmd,
                                     console=console,
                                     workflow_id=cmd.workflow_id,
                                     error_context=error_context.Copy())
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True, by_alias=True))

    url = client_cmd.lowda_url
    if url.endswith('/'):
      url = url[:-1]
    url += '/execute/'
    error_context['url'] = url
    logger.info('Executing remote workflow...',
                extra={'error_context': error_context.Dump()})

    res: ServerBase.ExecuteRes
    # res = await _ClientUploadAIOHTTPPost(url, req, error_context.Copy(), console)
    # res = await _ClientUploadHTTPXPost(url, req, error_context.Copy(), console)
    res = await _ClientUploadRequestsPost(action='execute workflow',
                                          url=url,
                                          req=req,
                                          resp_type=ServerBase.ExecuteRes,
                                          error_context=error_context.Copy(),
                                          console=console)
    error_context['res'] = res.model_dump(mode='json',
                                          round_trip=True,
                                          by_alias=True)
    logger.info('Executed remote workflow successfully.',
                extra={'error_context': error_context.Dump()})
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
  error_context = await _ErrorContext.Create(debug_path=await
                                             Path(cmd.debug_path).absolute(),
                                             key='Execute')
  workflow_id = uuid.uuid4().hex
  upload_req = await _ConstructUploadReq(upload_args=cmd,
                                         workflow_id=workflow_id,
                                         console=console,
                                         error_context=error_context.Copy())
  execute_req = await _ConstructExecuteReq(execute_args=cmd,
                                           workflow_id=workflow_id,
                                           console=console,
                                           error_context=error_context.Copy())
  provisioner = await GetProvisioner(cmd)
  remote = await GetFSSpecRemoteFileAPI(cmd.fs, console)
  server = await Server.Create(provisioner=provisioner,
                               remote=remote,
                               tmp_dir_path=await
                               Path(cmd.tmp_dir_path).absolute(),
                               debug_path=await Path(cmd.debug_path).absolute(),
                               debug_save_all=cmd.debug_save_all)
  upload_res = await server.UploadWorkflow(upload_req)
  if upload_res.error is not None:
    upload_error: ServerBase.UploadWorkflowError = upload_res.error
    console.print('Failed to upload workflow. Exiting.', style='red')
    console.print('Error:',
                  _YamlDump(
                      upload_error.model_dump(mode='json',
                                              round_trip=True,
                                              by_alias=True)),
                  style='red')
    console.print('Failed to upload workflow. Exiting.', style='red')
    return 1
  execute_res: ServerBase.ExecuteRes
  execute_res = await server.Execute(execute_req)
  if execute_res.error is not None:
    execute_error: ServerBase.ExecuteError = execute_res.error
    console.print('Failed to execute workflow. Exiting.', style='red')
    console.print('Error:',
                  _YamlDump(
                      execute_error.model_dump(mode='json',
                                               round_trip=True,
                                               by_alias=True)),
                  style='red')
    console.print('Failed to execute workflow. Exiting.', style='red')
    return 1
  if execute_res.success is None:
    raise AssertionError(
        'execute_res.success is None, and execute_res.error is None. This should not happen.'
    )
  execute_success: ServerBase.ExecuteSuccess = execute_res.success
  console.print('Workflow executed successfully.', style='green')
  console.print('Success:',
                _YamlDump(
                    execute_success.model_dump(mode='json',
                                               round_trip=True,
                                               by_alias=True)),
                style='green')
  console.print('Workflow executed successfully.', style='green')
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
