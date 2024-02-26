# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import asyncio
import os
import sys
import uuid

import fsspec
import yaml
from anyio import Path
from rich.console import Console
from slugify import slugify

from .lowda import (APIObjectInfo, APIWorkflow, ComfyRemoteInfo,
                     DumbProvisioner, FSSpecRemoteFileAPI, IOSpec,
                     LowdaOutputFieldType, LowdaOutputMapping, Manager,
                     ManagerBase, ProvisioningBundle, Workflow, WorkflowBundle,
                     WorkflowTemplateBundle)
from .comfyfs import RegisterComfyUIFS

async def amain():
  RegisterComfyUIFS()

  console = Console(file=sys.stderr)
  COMFY_API_URL: str | None = os.getenv('COMFY_API_URL')
  if COMFY_API_URL is None:
    raise ValueError('COMFY_API_URL is not set')

  comfy_api_url = COMFY_API_URL
  if not comfy_api_url.endswith('/'):
    comfy_api_url += '/'

  provisioner = DumbProvisioner(comfy_remote=ComfyRemoteInfo(
      comfy_api_url=COMFY_API_URL,
      upload={},
      download={
          'temp/': IOSpec(io_url=f'comfy+{comfy_api_url}temp/'),
          'output/': IOSpec(io_url=f'comfy+{comfy_api_url}output/'),
      },
  ))

  remote = FSSpecRemoteFileAPI(overwrite=FSSpecRemoteFileAPI.Overwrite.RENAME)
  remote.AddFS(uri_prefix='comfy+http://',
               fs=fsspec.filesystem('comfy+http'),
               mode='rw')
  remote.AddFS(uri_prefix='comfy+https://',
               fs=fsspec.filesystem('comfy+https'),
               mode='rw')

  manager = Manager(provisioner=provisioner,
                    remote=remote,
                    tmp_dir_path=await Path('.deleteme/tmp').absolute(),
                    debug_path=await Path('.deleteme/tmp').absolute(),
                    debug_save_all=True)

  provisioning = ProvisioningBundle(
      archives={},
      files={},
      custom_nodes={},
  )
  ############################################################################
  workflow_template = Workflow.model_validate_json(
      await Path('comfylowda/assets/sdxlturbo_example.json').read_text())
  api_workflow_template = APIWorkflow.model_validate_json(
      await Path('comfylowda/assets/sdxlturbo_example_api.json').read_text())
  object_info = APIObjectInfo.model_validate(
      yaml.safe_load(await
                     Path('comfylowda/assets/object_info.yml').read_text()))
  ############################################################################
  job_id = slugify(uuid.uuid4().hex)
  print('job_id:', job_id)
  workflow = WorkflowBundle(template_bundle=WorkflowTemplateBundle(
      workflow_template=workflow_template,
      api_workflow_template=api_workflow_template,
      important=[],
      object_info=object_info,
      user_input_mappings={},
      user_output_mappings={
          'Preview Image':
          LowdaOutputMapping(
              node_id='25',
              comfy_api_field_path='images[0]',
              comfy_api_field_type=LowdaOutputFieldType.TRIPLET_FILE_B64),
      }),
                            user_input_values={})
  res: ManagerBase.ExecuteRes
  res = await manager.Execute(
      Manager.ExecuteReq(job_id=job_id,
                         workflow=workflow,
                         provisioning=provisioning,
                         timeout=60))
  console.print(res)


asyncio.run(amain())
