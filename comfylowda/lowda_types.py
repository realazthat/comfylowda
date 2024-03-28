# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import enum
from pathlib import PurePath
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from comfy_catapult.comfy_schema import (APIHistoryEntry, APINodeID,
                                         APIObjectInfo, APIWorkflow,
                                         ComfyUIPathTriplet)
from pydantic import (BaseModel, ConfigDict, Field, RootModel, field_validator,
                      model_validator)
from typing_extensions import Annotated, Literal

from .comfy_schema import Workflow

JSON_SERIALIZABLE_TYPES = (str, int, float, bool, type(None), dict, list, tuple)


def IsJSONSerializable(value: Any) -> bool:
  if isinstance(value, dict):
    for k, v in value.items():
      if not IsJSONSerializable(k) or not IsJSONSerializable(v):
        return False
    return True
  if isinstance(value, (list, tuple)):
    for v in value:
      if not IsJSONSerializable(v):
        return False
    return True
  return isinstance(value, JSON_SERIALIZABLE_TYPES)


JSONSerializable = Annotated[Any, 'JSONSerializable', IsJSONSerializable]


def _URL(url: str) -> str:
  # TODO: This should check if the URL is valid; used for documentation.
  return url


class IOSpec(BaseModel):
  io_url: str
  """This could be a file URI, or any supported protocol.

  Additionally, it can be a comfy+http or comfy+https URL in the form of:
  comfy+https://comfy-server-host:port/folder_type/subfolder/sub/sub/filename

  This URL can be used to upload and download files to and from the comfy
  server.

  If the comfy+http or comfy+https URL is used, then the ComfyUI API will
  directly be used for upload and download for all upload/download operations.
  """

  @field_validator('io_url')
  @classmethod
  def check_io_url(cls, v):
    try:
      urlparse(v)
    except ValueError as e:
      raise ValueError(f'Invalid URL: {v}') from e
    return v

  kwargs: Dict[str, Any] = {}
  """kwargs to be passed to the fsspec.open() function.

  Useful for specifying account credentials.
  """


class ComfyRemoteInfo(BaseModel):
  comfy_api_url: str
  # {prefix => URL}
  upload: Dict[str, IOSpec]
  download: Dict[str, IOSpec]
  logs: IOSpec | None


class ProvisioningSpec(BaseModel):
  files: Dict[str, IOSpec]
  archives: Dict[str, IOSpec]
  custom_nodes: Dict[str, IOSpec]


class InputPPKind(str, enum.Enum):
  VALUE = 'VALUE'
  """Anything JSON Serializable. Just gets passed through."""
  FILE = 'FILE'
  """A file. Gets uploaded according to the upload spec."""


class UserIOSpec(RootModel[IOSpec | str]):
  root: IOSpec | str

  @field_validator('root')
  @classmethod
  def check_root(cls, v):
    if isinstance(v, str):
      try:
        urlparse(v)
      except ValueError as e:
        raise ValueError(f'Invalid URL: {v}') from e
    return v

  def ToIOSpec(self) -> IOSpec:
    if not isinstance(self.root, str):
      return self.root
    return IOSpec(io_url=self.root)


class FileUploadMapSpec(BaseModel):
  upload_to: str | ComfyUIPathTriplet = Field(
      ...,
      description="""Location to upload the file to.

Either a path relative to ComfyUI installation, or a triplet.

The path must not be absolute.

Note that some nodes only allow certain subfolders, e.g 'temp' or 'output/custom_node_name'.""",
      alias='to')

  @field_validator('upload_to')
  @classmethod
  def check_upload_to(cls, v):
    if isinstance(v, str):
      if PurePath(v).is_absolute():
        raise ValueError(f'upload_to must not be absolute: {v}')
    return v

  node_mode: Literal['TRIPLET', 'FILEPATH'] = Field(
      ...,
      description="""How the node expects the filepath to be passed in as input.

Some nodes expect a triplet, some expect a relative filepath string.

For those that expect a relative filepath string, some expect it relative to
the ComfyUI installation path, and some expect it relative to the 'input'
folder; others use other relative paths, such as 'input/custom_node_inputs/'
or even in some non-standard relative path (this is only supported if the
provisioned ComfyUI instance is specified with a way to upload files other
than the ComfyUI API, because the ComfyUI API only supports upload to
`/path/to/comfyui/input`).

See discard_prefix.
""",
      alias='mode')
  discard_prefix: str | None = Field(
      ...,
      description=
      """The prefix to be discarded from the filepath when passing the filepath to the node.

This should only be used with FILEPATH mode, and only if node expects filepath
strings to be relative to something other than the ComfyUI root path.

Some nodes assume that a filepath is relative to a certain path.

For example, the 'Load Image' node assumes that the filepath is relative to
the 'input' folder. So the prefix for that should be 'input/'.

This prefix should not start with a slash.

Note on trailing slash: different nodes may or may not require a trailing
slash here; depending on if the node expects a slash in the beginning of the
file path.
""",
      alias='pfx')

  @field_validator('discard_prefix')
  @classmethod
  def check_discard_prefix(cls, v):
    if v is not None:
      if not isinstance(v, str):
        raise ValueError(f'discard_prefix must be str or None: {v}')
      if PurePath(v).is_absolute():
        raise ValueError(f'discard_prefix must not be absolute: {v}')
    return v

  @model_validator(mode='after')
  def check_prefix_implies_filename(self):
    prefix_implies_filename = (self.discard_prefix is None
                               or self.node_mode == 'FILEPATH')
    if not prefix_implies_filename:
      raise ValueError(
          'discard_prefix is only valid when node_mode is FILEPATH')
    return self


class FileDownloadMapSpec(BaseModel):
  """If this is set, then a node output will be interpreted as a file and downloaded according to a user specified spec.

  This class specifies how to extract the file from the node output and download it.

  The user must specify an input value to specify the download URL to their system.

  They can either use a globally accessible IOSpec, or a string 'base64'.

  TODO: Show example.
  """
  node_mode: Literal['TRIPLET', 'FILEPATH'] = Field(
      ...,
      description=
      f"""How the node output (see {_URL("https://github.com/realazthat/comfylowda/blob/master/comfylowda/assets/example_history.yaml")}) outputs the filepath.

Some nodes output a triplet, some output a filepath, some output a filepath that is relative to a certain path inside the ComfyUI installation.
""",
      alias='mode')
  prepend_prefix: str | None = Field(
      ...,
      description=
      """The prefix to be prepended to the filepath reading the value from the node.

Some nodes assume that a filepath is relative to a certain path.

This tells the downloader to prepend the prefix in order to find the file.
  """,
      alias='pfx')

  @field_validator('prepend_prefix')
  @classmethod
  def check_prepend_prefix(cls, v):
    if v is not None:
      if not isinstance(v, str):
        raise ValueError(f'prepend_prefix must be str or None: {v}')
      if PurePath(v).is_absolute():
        raise ValueError(f'prepend_prefix must not be absolute: {v}')
    return v

  @model_validator(mode='after')
  def check_exclusivity(self):
    prefix_implies_filename = (self.prepend_prefix is None
                               or self.node_mode == 'FILEPATH')
    if not prefix_implies_filename:
      raise ValueError(
          'prepend_prefix is only valid when node_mode is FILEPATH')
    return self


class OutputPPKind(str, enum.Enum):
  NODE_VALUE = 'NODE_VALUE'
  """Anything JSON Serializable. Just gets passed through as an output value."""
  FILE = 'FILE'
  """JSON Serializable dict that will be validated using DownloadFile."""


class InputMapping(BaseModel):
  name: str = Field(
      ...,
      description=
      'The name of the input. This is the key that the user will be expected to use in their input JSON.'
  )
  node: APINodeID | str | int = Field(
      ...,
      description=
      'The id or unique title of a node in the ComfyUI {API, regular} Workflow.'
  )
  field: str | None = Field(
      ...,
      description=
      f"""A pydash field path, for the pydash.get() and pydash.set_() functions.

The field_path begins at the .inputs field of a node in the ComfyUI Workflow API format.'
See {_URL('https://github.com/realazthat/comfylowda/blob/master/comfylowda/assets/sdxlturbo_example_api.json')}
for an example Workflow API format. If specified, this will copy the'
input value to the field path.
""")
  pp: InputPPKind = Field(
      ..., description='You can choose a preprocessor, e.g upload a file.')
  spec: FileUploadMapSpec | None = Field(
      None, description='If pp!=VALUE, then this is required.', alias='spec')

  @model_validator(mode='after')
  def check_exclusivity(self):
    if (self.pp != InputPPKind.VALUE) != (self.spec is not None):
      raise ValueError(
          'If pp!=VALUE, then spec is required. Otherwise, it must be None.')
    return self

  user_json_spec: Literal['NO_PUBLIC_INPUT'] | Literal['ANY'] | Literal[
      'OPT_ANY'] | dict = Field(
          ..., description='JSON spec to verify the user input.')
  user_value: JSONSerializable = Field(
      ..., description='The default input value to use.')


class OutputMapping(BaseModel):
  """Defines a mapping from the /history/{prompt_id} entry to a field in the user output.
  """
  model_config = ConfigDict(use_enum_values=True)
  name: str = Field(
      ...,
      description="""
The name of the output. This is the key that will be used in the output JSON.

If this mapping requires user input, this is the key that the user will be
expected to use in their output JSON.
""")
  node: APINodeID | str | int = Field(
      ...,
      description=
      'The id or unique title of a node in the ComfyUI {API, regular} Workflow')
  field: str = Field(
      ...,
      description=
      'A pydash field path, for the pydash.get() and pydash.set_() functions.'
      ' The field_path begins at the job_prompt_id.outputs.node_id in the ComfyUI Workflow API format.'
      ' See https://github.com/realazthat/comfylowda/blob/master/comfylowda/assets/example_history.yaml'
      ' for an example ComfyUI /history format.')
  pp: OutputPPKind = Field(
      ...,
      description=
      'You can choose a field type to have lowda do some post processing, e.g download a file.'
  )
  spec: FileDownloadMapSpec | None = Field(
      ...,
      description='If pp!=NODE_VALUE, then this is required.',
      alias='spec')
  user_json_spec: Literal['NO_PUBLIC_INPUT'] | Literal['ANY'] | Literal[
      'OPT_ANY'] | dict = Field(
          ..., description='JSON spec to verify the user input.')
  user_value: JSONSerializable = Field(
      ..., description='The default input value to use.')

  @model_validator(mode='after')
  def check_exclusivity(self):
    if (self.pp != OutputPPKind.NODE_VALUE) != (self.spec is not None):
      raise ValueError(
          'If pp!=NODE_VALUE, then spec is required. Otherwise, it must be None.'
      )
    return self


class WorkflowTemplateBundle(BaseModel):
  workflow_template: Workflow
  api_workflow_template: APIWorkflow
  important: List[APINodeID]
  object_info: APIObjectInfo
  input_mappings: List[InputMapping] = Field(
      ...,
      description=
      'Input mappings. These define the inputs to the workflow. Names should be unique, and also should not collide with output names.'
  )
  output_mappings: List[OutputMapping] = Field(
      ...,
      description=
      'Output mappings. These define the outputs of the workflow. Names should be unique, and also should not collide with input names.'
  )

  @model_validator(mode='after')
  def check_input_mappings(self):
    iname2idx = {}
    for i, input_mapping in enumerate(self.input_mappings):
      if input_mapping.name in iname2idx:
        prev_idx = iname2idx[input_mapping.name]
        raise ValueError(
            f'Duplicate input name: {input_mapping.name}, at index {i} and {prev_idx}'
        )
      iname2idx[input_mapping.name] = i
    oname2idx = {}
    for i, output_mapping in enumerate(self.output_mappings):
      if output_mapping.name in iname2idx:
        prev_idx = iname2idx[output_mapping.name]
        raise ValueError(
            f'Duplicate input name: {output_mapping.name}, at index {i} and {prev_idx}'
        )
      if output_mapping.name in oname2idx:
        prev_idx = oname2idx[output_mapping.name]
        raise ValueError(
            f'Duplicate output name: {output_mapping.name}, at index {i} and {prev_idx}'
        )
      oname2idx[output_mapping.name] = i
    return self


class UploadWorkflowReq(BaseModel):
  workflow_id: str
  template_bundle: WorkflowTemplateBundle
  prov_spec: ProvisioningSpec


class UploadWorkflowError(BaseModel):
  error_id: str = Field(..., description='The error id.')
  status_code: Optional[int] = Field(
      ..., description='The error status code. To be used in HTTP responses.')
  name: str = Field(..., description='The error name.')
  message: str = Field(..., description='The error message.')
  context: dict = Field(..., description='The error context.')


class UploadWorkflowRes(BaseModel):
  error: Optional[UploadWorkflowError]


class DownloadWorkflowReq(BaseModel):
  workflow_id: str


class DownloadWorkflowSuccess(BaseModel):
  workflow_id: str
  template_bundle: WorkflowTemplateBundle
  prov_spec: ProvisioningSpec


class DownloadWorkflowError(BaseModel):
  error_id: str = Field(..., description='The error id.')
  status_code: Optional[int] = Field(
      ..., description='The error status code. To be used in HTTP responses.')
  name: str = Field(..., description='The error name.')
  message: str = Field(..., description='The error message.')
  context: dict = Field(..., description='The error context.')


class DownloadWorkflowRes(BaseModel):
  success: Optional[DownloadWorkflowSuccess]
  error: Optional[DownloadWorkflowError]


class ExecuteReq(BaseModel):
  job_id: str = Field(
      ...,
      description=
      'The job id. This is user supplied, should be unique or bad things can happen.'
  )
  workflow_id: str = Field(
      ..., description='The workflow id. This must match an uploaded workflow.')
  user_input_values: Dict[str, JSONSerializable] = Field(
      ...,
      description=
      'The user input values, as per the InputMappings and OutputMapping specs of the uploaded workflow.'
  )
  keepalive: float = Field(
      ...,
      description=
      'The keepalive time in seconds. This will be used to keep the job alive, the files it is depending on etc. for as long as the timeout. The client should periodically extend the timeout as long as it is alive, using the touch mechanism.'
  )


class ExecuteSuccess(BaseModel):
  job_id: str = Field(..., description='The job id.')
  history: APIHistoryEntry = Field(..., description='The history entry.')
  mapped_outputs: Dict[str, JSONSerializable] = Field(
      ...,
      description=
      'The outputs, as per the OutputMappings specs of the uploaded workflow.')


class ExecuteError(BaseModel):
  error_id: str = Field(..., description='The error id.')
  status_code: Optional[int] = Field(
      ..., description='The error status code. To be used in HTTP responses.')
  name: str = Field(..., description='The error name.')
  message: str = Field(..., description='The error message.')
  context: dict = Field(..., description='The error context.')


class ExecuteRes(BaseModel):
  success: Optional[ExecuteSuccess]
  error: Optional[ExecuteError]


class TouchReq(BaseModel):
  job_id: str = Field(..., description='The job id to keep alive.')
  keepalive: float = Field(..., description='The keepalive time in seconds.')


class TouchRes(BaseModel):
  success: bool = Field(..., description='Whether the touch was successful.')
  message: str = Field(..., description='A message, maybe useful in failure.')
