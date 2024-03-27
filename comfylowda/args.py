# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import argparse
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator
from rich_argparse import RichHelpFormatter  # type: ignore[import]

from .pydantic_args import ArgConfigDict, ParserHelper


class FSArg(BaseModel):
  arg_config: ClassVar[ArgConfigDict] = ArgConfigDict(parse_as='STRING')

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


class ServeAppArgs(BaseModel):
  bind: str
  port: int


class ComfyApiURLEnv(BaseModel):
  COMFY_API_URL: str | None

  @field_validator('COMFY_API_URL')
  @classmethod
  def COMFY_API_URL_is_url(cls, v):
    if v is None:
      return v
    try:
      urlparse(v)
    except Exception as e:
      raise ValueError(f'Invalid URL: {v}') from e
    return v


DEFAULT_COMFY_API_URL = ComfyApiURLEnv.model_validate(
    {'COMFY_API_URL': os.getenv('COMFY_API_URL', None)})


class ProvisionerArgs(BaseModel):
  comfy_api_url: str = Field(
      DEFAULT_COMFY_API_URL.COMFY_API_URL or ...,
      validate_default=True,
      description=
      'URL of the Comfy API server. Defaults to environment variable COMFY_API_URL.'
  )

  @field_validator('comfy_api_url')
  @classmethod
  def comfy_api_url_is_url(cls, v):
    if v is None:
      raise ValueError('--comfy_api_url is not set, and COMFY_API_URL is not')
    try:
      urlparse(v)
    except Exception as e:
      raise ValueError(f'Invalid URL: {v}') from e
    return v


class DebugEnv(BaseModel):
  COMFYLOWDA_DEBUG: Literal['0', '1']


COMFYLOWDA_DEBUG = DebugEnv.model_validate(
    {'COMFYLOWDA_DEBUG': os.getenv('COMFYLOWDA_DEBUG', '0')})


class CommonArgs(BaseModel):
  debug: bool = Field(
      COMFYLOWDA_DEBUG.COMFYLOWDA_DEBUG == '1',
      description=
      'Enable debug mode, defaults to environment variable COMFYLOWDA_DEBUG∈{0,1}.'
  )
  debug_path: Path = Field(
      '.logs/debug',
      description='Location to store large files too big for the logs.')
  log_to_stderr: bool = Field(
      False,
      description='Log to stderr. Defaults to False. Useful for debugging.')
  log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
      'INFO',
      description=
      "Log level. Defaults to INFO. Useful for debugging. Example: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'."
  )
  json_log_path: Path | None = Field(
      '.logs/comfylowda.json.log',
      description=
      "Path to the JSON log file. Defaults to '.logs/comfylowda.json.log'.")
  yaml_log_path: Path | None = Field(
      '.logs/comfylowda.yaml.log',
      description=
      "Path to the YAML log file. Defaults to '.logs/comfylowda.yaml.log'.")

  fs: List[FSArg] = Field(
      [], description='Filesystems (local/and remote) to register.')


class ServerArgs(BaseModel):
  tmp_dir_path: Path
  debug_save_all: bool


class ServeCommand(ServerArgs, ProvisionerArgs, ServeAppArgs, CommonArgs):
  arg_config: ClassVar[ArgConfigDict] = ArgConfigDict(parse_as='COMMAND')


################################################################################
class UploadArgs(BaseModel):
  workflow: Path = Field(..., description='Path to the workflow to upload.')
  api_workflow: Path = Field(...,
                             description='Path to the API workflow to upload.')
  object_info: Path = Field(...,
                            description='Path to the object info to upload.')
  im: List[str] = Field([], description='JSON string of an InputMapping')
  om: List[str] = Field([], description='JSON string of an OutputMapping')


class ClientUploadCommand(UploadArgs, BaseModel):
  arg_config: ClassVar[ArgConfigDict] = ArgConfigDict(parse_as='COMMAND')
  workflow_id: str = Field(..., description='ID of the workflow to upload.')


class InputNameArg(BaseModel):
  arg_config: ClassVar[ArgConfigDict] = ArgConfigDict(parse_as='STRING')
  name: str = Field(..., description='Name of the input')
  json_value: Any = Field(..., description='Value of the input in JSON format')


class ExecuteArgs(BaseModel):
  i: List[InputNameArg] = Field([], description='Input value')
  keepalive: float = Field(
      60.0,
      description='Keepalive interval for the connection to the server, in'
      ' seconds. The client will send a keepalive message to the server every'
      ' keepalive seconds. Defaults to 60.0 seconds.')


class ClientExecuteCommand(ExecuteArgs, BaseModel):
  arg_config: ClassVar[ArgConfigDict] = ArgConfigDict(parse_as='COMMAND')
  workflow_id: str = Field(..., description='ID of the workflow to execute')


class ClientCommand(CommonArgs, BaseModel):
  arg_config: ClassVar[ArgConfigDict] = ArgConfigDict(parse_as='COMMAND')

  lowda_url: str = Field(..., description='URL of the Lowda server.')
  upload: Optional[ClientUploadCommand] = Field(
      description='Upload a workflow to the server.')
  execute: Optional[ClientExecuteCommand] = Field(
      description='Execute a workflow on the server.')


class ExecuteCommand(ProvisionerArgs, ExecuteArgs, UploadArgs, ServerArgs,
                     CommonArgs, BaseModel):
  """Combines serve, client upload, client execute commands in one, to execute a single workflow once."""
  arg_config: ClassVar[ArgConfigDict] = ArgConfigDict(parse_as='COMMAND')


class Arguments(BaseModel):

  serve: Optional[ServeCommand] = Field(
      description='Start a server to allow workflows to be uploaded, and served'
  )
  client: Optional[ClientCommand] = Field(
      description=
      'Connect to a server to upload a workflow to be served or to execute a workflow that is being served.'
  )
  execute: Optional[ExecuteCommand] = Field(
      description=
      'Execute a workflow directly, no lowda client/server necessary (still need a comfyui instance somewhere).'
  )


def Parse(*, version: str) -> Arguments:
  error_context: Dict[str, Any] = {}
  parser = argparse.ArgumentParser(
      prog='Comfylowda',
      description='Productionizing ComfyUI Workflows',
      formatter_class=RichHelpFormatter,
  )
  parser.add_argument('--version', action='version', version=version)

  helper = ParserHelper()
  context = ParserHelper.Context(append_or_nargs='REPEAT',
                                 parse_as='ENUMERATE',
                                 wire_format='JSON',
                                 subparsers=None)
  helper._AddType(parser=parser,
                  id='root',
                  parent_dest=None,
                  name_or_flags='N/A',
                  metavar='N/A',
                  t=Arguments,
                  description=Arguments.__doc__,
                  is_required=True,
                  default=None,
                  context=context,
                  add_args_kwargs=None,
                  error_context=error_context.copy())
  args = parser.parse_args()

  print(args)
  # return args
  args_typed: Arguments = helper.Parse(args, Arguments, context)
  print(args_typed)

  return args_typed
