# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import List, Literal

import fsspec
from anyio import Path
from comfy_catapult.comfy_schema import APIObjectInfo, APIWorkflow
from pydantic import BaseModel, Field
from rich.console import Console
from slugify import slugify

from .comfy_schema import Workflow
from .comfyfs import RegisterComfyUIFS
from .lowda import (ComfyRemoteInfo, DumbProvisioner, FileDownloadMapSpec,
                    FileUploadMapSpec, FSSpecRemoteFileAPI, InputMapping,
                    InputPPKind, IOSpec, Manager, ManagerBase, OutputMapping,
                    OutputPPKind, ProvisioningBundle, UserIOSpec,
                    WorkflowBundle, WorkflowTemplateBundle)


async def amain():

  console = Console(file=sys.stderr)
  try:
    RegisterComfyUIFS()

    parser = argparse.ArgumentParser(description='Comfylowda')
    parser.add_argument('--workflow',
                        type=Path,
                        required=True,
                        help='Workflow file path')
    parser.add_argument('--api-workflow',
                        type=Path,
                        required=True,
                        help='API Workflow file path')
    parser.add_argument('--object-info',
                        type=Path,
                        required=True,
                        help='API Object Info file path')
    parser.add_argument('--comfy-api-url',
                        type=str,
                        default=None,
                        help='Comfy API URL')
    parser.add_argument('--tmp-dir-path',
                        type=Path,
                        default=Path('.deleteme/tmp'),
                        help='Temporary directory path')
    parser.add_argument('--debug-path',
                        type=Path,
                        default=Path('.deleteme/debug'),
                        help='Debug directory path')
    parser.add_argument('--debug-save-all',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Save all debug data (default: False)')
    parser.add_argument('--keepalive-timeout',
                        type=float,
                        default=60.0,
                        help='Keepalive timeout (default: 60.0)')

    class FSArg(BaseModel):
      prefix: str = Field(
          ...,
          description=
          'URL Prefix to register for this filesystem. Example: "file:///path/to/dir/" or "https://huggingface.co/"',
          alias='pfx')
      protocol: str = Field(
          ...,
          description=
          'The protocol for fsspec. Example: "file", or "comfy+http" or "comfy+https" or "https"',
          alias='proto')
      mode: Literal['r', 'w', 'rw'] = Field(
          'rw', description='The mode for fsspec. Example: "r", "w", "rw".')

    parser.add_argument(
        '-fs',
        type=FSArg.model_validate_json,
        action='append',
        default=[],
        help=
        'Add a filesystem. Example: {"prefix": "file:///path/to/dir/", "protocol": "file", "mode": "rw"}.'
    )

    input_mapping_example = InputMapping(
        name='Load Image',
        node='Load Image',
        field='image',
        pp=InputPPKind.FILE,
        spec=FileUploadMapSpec(to='input/MyImage.png', mode='TRIPLET',
                               pfx=None),
        user_json_spec=UserIOSpec.model_json_schema(),
        user_value=None)
    ouput_mapping_example = OutputMapping(name='Preview Image',
                                          node=25,
                                          field='images[0]',
                                          pp=OutputPPKind.FILE,
                                          spec=FileDownloadMapSpec(
                                              mode='TRIPLET', pfx=None),
                                          user_json_spec='NO_PUBLIC_INPUT',
                                          user_value='base64')

    parser.add_argument(
        '-im',
        type=str,
        action='append',
        default=[],
        help=
        f'Input mappings. Example: {input_mapping_example.model_dump_json()}')
    parser.add_argument(
        '-om',
        type=str,
        action='append',
        default=[],
        help=
        f'Output mappings. Example: {ouput_mapping_example.model_dump_json()}')
    parser.add_argument(
        '-i',
        type=str,
        action='append',
        default=[],
        help=
        'User input values. Example: {"name": "MyImage.png", "type": "image/png"}'
    )
    args = parser.parse_args()

    ##############################################################################
    comfy_api_url: str | None = args.comfy_api_url
    if comfy_api_url is None:
      comfy_api_url = os.getenv('COMFY_API_URL')
      if comfy_api_url is None:
        raise ValueError('COMFY_API_URL is not set')
    ##############################################################################
    tmp_dir_path: Path = args.tmp_dir_path
    tmp_dir_path = await tmp_dir_path.absolute()
    debug_path: Path = args.debug_path
    debug_path = await debug_path.absolute()
    debug_save_all: bool = args.debug_save_all
    ##############################################################################
    workflow_path: Path = args.workflow
    api_workflow_path: Path = args.api_workflow
    object_info_path: Path = args.object_info
    ##############################################################################
    keepalive_timeout: float = args.keepalive_timeout

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

    remote = FSSpecRemoteFileAPI(overwrite=FSSpecRemoteFileAPI.Overwrite.RENAME)
    remote.AddFS(url_prefix='comfy+http://',
                 fs=fsspec.filesystem('comfy+http'),
                 mode='rw')
    remote.AddFS(url_prefix='comfy+https://',
                 fs=fsspec.filesystem('comfy+https'),
                 mode='rw')
    ############################################################################
    fs_arg: FSArg
    for fs_arg in args.fs:
      remote.AddFS(url_prefix=fs_arg.prefix,
                   fs=fsspec.filesystem(fs_arg.protocol),
                   mode=fs_arg.mode)
    ############################################################################
    manager = Manager(provisioner=provisioner,
                      remote=remote,
                      tmp_dir_path=tmp_dir_path,
                      debug_path=debug_path,
                      debug_save_all=debug_save_all)

    provisioning = ProvisioningBundle(
        archives={},
        files={},
        custom_nodes={},
    )
    ############################################################################
    workflow_template = Workflow.model_validate_json(await
                                                     workflow_path.read_text())
    api_workflow_template = APIWorkflow.model_validate_json(
        await api_workflow_path.read_text())
    object_info = APIObjectInfo.model_validate_json(
        await object_info_path.read_text())
    ############################################################################
    input_mappings: List[InputMapping] = []
    for im in args.im:
      input_mappings.append(InputMapping.model_validate_json(im))
    output_mappings: List[OutputMapping] = []
    for om in args.om:
      output_mappings.append(OutputMapping.model_validate_json(om))
    user_input_values = {}
    input_str: str
    for input_str in args.i:
      input_dict = json.loads(input_str)
      name = input_dict['name']
      user_input_values[name] = input_dict['value']
    ############################################################################
    job_id = slugify(uuid.uuid4().hex)
    template_bundle = WorkflowTemplateBundle(
        workflow_template=workflow_template,
        api_workflow_template=api_workflow_template,
        important=[],
        object_info=object_info,
        input_mappings=input_mappings,
        output_mappings=output_mappings)

    workflow = WorkflowBundle(template_bundle=template_bundle,
                              user_input_values=user_input_values)
    res: ManagerBase.ExecuteRes
    res = await manager.Execute(
        Manager.ExecuteReq(job_id=job_id,
                           workflow=workflow,
                           provisioning=provisioning,
                           keepalive=keepalive_timeout))
    console.print(res)
  except Exception:
    console.print_exception()
    console.print('Failed to execute the workflow. Exiting.', style='red')
    sys.exit(1)


if __name__ == '__main__':
  asyncio.run(amain())
