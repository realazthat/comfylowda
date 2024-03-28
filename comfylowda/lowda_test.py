# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import asyncio
import json
import logging
import os
import sys
import uuid

import fsspec
import yaml
from anyio import Path
from comfy_catapult.comfy_schema import APIObjectInfo, APIWorkflow
from rich.console import Console
from slugify import slugify

from .comfy_schema import Workflow
from .comfyfs import RegisterComfyUIFS
from .error_utils import _YamlDump
from .lowda import DumbProvisioner, FSSpecRemoteFileAPI, Server
from .lowda_types import (ComfyRemoteInfo, FileDownloadMapSpec, InputMapping,
                          InputPPKind, IOSpec, OutputMapping, OutputPPKind,
                          ProvisioningSpec, WorkflowTemplateBundle)


class JSONFormatter(logging.Formatter):

  def format(self, record):
    record.message = record.getMessage()
    if self.usesTime():
      record.asctime = self.formatTime(record, self.datefmt)
    # Include any extra attributes in the log record, such as extra parameters
    return json.dumps(record.__dict__)


class YAMLFormatter(logging.Formatter):

  def format(self, record):
    record.message = record.getMessage()
    if self.usesTime():
      record.asctime = self.formatTime(record, self.datefmt)
    # Include any extra attributes in the log record, such as extra parameters
    return '---\n' + _YamlDump(record.__dict__)


async def amain():

  console = Console(file=sys.stderr)
  try:
    ############################################################################
    json_log_path = await Path('.deleteme/.logs/lowda_test_log.json.log'
                               ).absolute()
    await json_log_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_log_path = await Path('.deleteme/.logs/lowda_test_log.yaml.log'
                               ).absolute()
    await yaml_log_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path: Path = await Path('.deleteme/debug').absolute()
    await debug_path.mkdir(parents=True, exist_ok=True)
    tmp_dir_path: Path = await Path('.deleteme/tmp').absolute()
    await tmp_dir_path.mkdir(parents=True, exist_ok=True)
    object_info_path = await Path('comfylowda/assets/object_info.yml'
                                  ).absolute()
    workflow_path = await Path('comfylowda/assets/sdxlturbo_example.json'
                               ).absolute()
    api_workflow_path = await Path(
        'comfylowda/assets/sdxlturbo_example_api.json').absolute()
    local_download_path = await Path('.deleteme/lowda_test/local_download'
                                     ).absolute()
    ############################################################################

    json_file_handler = logging.FileHandler(str(json_log_path))
    json_file_handler.setFormatter(JSONFormatter())
    yaml_file_handler = logging.FileHandler(str(yaml_log_path))
    yaml_file_handler.setFormatter(YAMLFormatter())
    logging.basicConfig(handlers=[json_file_handler, yaml_file_handler],
                        level=logging.INFO)

    RegisterComfyUIFS()
    COMFY_API_URL: str | None = os.getenv('COMFY_API_URL')
    if COMFY_API_URL is None:
      raise ValueError('COMFY_API_URL is not set')

    comfy_api_url = COMFY_API_URL
    if not comfy_api_url.endswith('/'):
      comfy_api_url += '/'

    provisioner = DumbProvisioner(comfy_remote=ComfyRemoteInfo(
        comfy_api_url=comfy_api_url,
        upload={},
        download={
            'temp/': IOSpec(io_url=f'comfy+{comfy_api_url}temp/'),
            'output/': IOSpec(io_url=f'comfy+{comfy_api_url}output/'),
        },
        logs=None,
    ))

    remote = FSSpecRemoteFileAPI(overwrite=FSSpecRemoteFileAPI.Overwrite.RENAME)
    remote.AddFS(url_prefix='comfy+http://',
                 fs=fsspec.filesystem('comfy+http'),
                 mode='rw')
    remote.AddFS(url_prefix='comfy+https://',
                 fs=fsspec.filesystem('comfy+https'),
                 mode='rw')
    remote.AddFS(url_prefix=local_download_path.as_uri(),
                 fs=fsspec.filesystem('file'),
                 mode='w')

    manager = await Server.Create(provisioner=provisioner,
                                  remote=remote,
                                  tmp_dir_path=tmp_dir_path,
                                  debug_path=debug_path,
                                  debug_save_all=True)

    provisioning = ProvisioningSpec(
        archives={},
        files={},
        custom_nodes={},
    )
    ############################################################################
    workflow_template = Workflow.model_validate_json(await
                                                     workflow_path.read_text())
    api_workflow_template = APIWorkflow.model_validate_json(
        await api_workflow_path.read_text())
    object_info = APIObjectInfo.model_validate(
        yaml.safe_load(await object_info_path.read_text()))
    ############################################################################
    job_id = slugify(uuid.uuid4().hex)
    print('job_id:', job_id)

    template_bundle = WorkflowTemplateBundle(
        workflow_template=workflow_template,
        api_workflow_template=api_workflow_template,
        important=[],
        object_info=object_info,
        input_mappings=[
            InputMapping(name='width',
                         node='EmptyLatentImage',
                         field='width',
                         pp=InputPPKind.VALUE,
                         spec=None,
                         user_json_spec='ANY',
                         user_value=None),
            InputMapping(name='height',
                         node='EmptyLatentImage',
                         field='width',
                         pp=InputPPKind.VALUE,
                         spec=None,
                         user_json_spec='ANY',
                         user_value=None),
        ],
        output_mappings=[
            OutputMapping(name='Preview Image',
                          node='25',
                          field='images[0]',
                          pp=OutputPPKind.FILE,
                          spec=FileDownloadMapSpec(mode='TRIPLET', pfx=None),
                          user_json_spec='ANY',
                          user_value=None),
        ])
    workflow_id = slugify(uuid.uuid4().hex)

    await local_download_path.mkdir(parents=True, exist_ok=True)
    download_io_specs = [
        'base64', (local_download_path / 'done.png').as_uri(),
        IOSpec(io_url=(local_download_path /
                       'done2.png').as_uri()).model_dump(mode='json')
    ]
    for download_io_spec in download_io_specs:
      user_input_values = {
          'width': 512,
          'height': 512,
          'Preview Image': download_io_spec
      }
      ##########################################################################
      res: Server.UploadWorkflowRes
      res = await manager.UploadWorkflow(
          Server.UploadWorkflowReq(workflow_id=workflow_id,
                                   template_bundle=template_bundle,
                                   prov_spec=provisioning))
      console.print(res)
      ##########################################################################
      exec_res: Server.ExecuteRes
      exec_res = await manager.Execute(
          Server.ExecuteReq(job_id=job_id,
                            workflow_id=workflow_id,
                            user_input_values=user_input_values,
                            keepalive=60.))
      console.print(exec_res)
      ##########################################################################
  except Exception as e:
    console.print_exception()
    raise e


if __name__ == '__main__':
  asyncio.run(amain())
