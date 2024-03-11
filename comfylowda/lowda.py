# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import asyncio
import base64
import enum
import json
import logging
import textwrap
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import PurePath
from typing import Annotated, Any, Dict, List, Literal, Tuple
from urllib.parse import urlparse

import aiofiles
import fsspec
import jsonschema
import pydantic
import pydash
import yaml
from anyio import Path
from comfy_catapult.api_client import ComfyAPIClient
from comfy_catapult.catapult import ComfyCatapult
from comfy_catapult.comfy_schema import (APIHistoryEntry, APINodeID,
                                         APIObjectInfo, APIOutputUI,
                                         APIWorkflow, APIWorkflowNodeInfo,
                                         ComfyFolderType, ComfyUIPathTriplet)
from comfy_catapult.comfy_utils import WatchVar
from datauri import DataURI
from pydantic import (BaseModel, ConfigDict, Field, RootModel, field_validator,
                      model_validator)
from slugify import slugify

from .comfy_schema import Workflow, WorkflowNode
from .comfyfs import _Writable

logger = logging.getLogger(__name__)

JSON_SERIALIZABLE_TYPES = (str, int, float, bool, type(None), dict, list, tuple)


def _IsJSONSerializable(value: Any) -> bool:
  if isinstance(value, dict):
    for k, v in value.items():
      if not _IsJSONSerializable(k) or not _IsJSONSerializable(v):
        return False
    return True
  if isinstance(value, (list, tuple)):
    for v in value:
      if not _IsJSONSerializable(v):
        return False
    return True
  return isinstance(value, JSON_SERIALIZABLE_TYPES)


JSONSerializable = Annotated[Any, 'JSONSerializable', _IsJSONSerializable]


class _CustomDumper(yaml.Dumper):

  def represent_tuple(self, data):
    return self.represent_list(data)


_CustomDumper.add_representer(tuple, _CustomDumper.represent_tuple)


def YamlDump(data: Any) -> str:
  return yaml.dump(data, indent=2, Dumper=_CustomDumper, sort_keys=False)


class _ErrorContext:

  def __init__(self, *, debug_path: Path, key: str):
    self._debug_path = debug_path
    self._key = key
    self._error_context_path = self._debug_path / slugify(self._key)
    self._context: Dict[str, JSONSerializable] = {}

  async def LargeToFile(self, name: str, msg: JSONSerializable) -> str:
    directory = self._error_context_path / slugify(datetime.now().isoformat())
    await directory.mkdir(parents=True, exist_ok=True)
    path = directory / slugify(name)
    yaml_path = path.with_suffix('.yaml')
    await yaml_path.write_text(YamlDump(msg))
    return yaml_path.as_uri()

  def Add(self, *, name: str, msg: JSONSerializable) -> None:
    self._context[name] = msg

  def Dump(self) -> JSONSerializable:
    return self._context

  def Copy(self) -> '_ErrorContext':
    copy_cxt = _ErrorContext(debug_path=self._debug_path, key=self._key)
    copy_cxt._context = self._context.copy()
    return copy_cxt

  def __setitem__(self, key: str, value: JSONSerializable) -> None:
    self._context[key] = value


def _URL(url: str) -> str:
  # TODO: This should check if the URL is valid; used for documentation.
  return url


def _DiscardPrefix(path: str, prefix: str | None) -> str:
  if prefix is None:
    return path
  if not path.startswith(prefix):
    raise ValueError(f'Expected {path} to start with {prefix}')
  return path[len(prefix):]


def _PathToTriplet(path: PurePath) -> ComfyUIPathTriplet:
  # TODO: Migrate this back to comfy_catapult, and also decompose
  # ComfySchemeURLToTriplet() to use this.

  if path.is_absolute():
    raise ValueError(
        f'path => (folder_type, subfolder, filename) must not be absolute: {path}'
    )

  if len(path.parts) < 3:
    raise ValueError(
        f'path => (folder_type, subfolder, filename) must have at least 3 parts: {path}'
    )

  folder_type_validator = pydantic.TypeAdapter[ComfyFolderType](ComfyFolderType)
  folder_type: ComfyFolderType = folder_type_validator.validate_python(
      path.parts[0])
  subfolder = path.parts[1]
  filename = path.parts[2]
  return ComfyUIPathTriplet(type=folder_type,
                            subfolder=subfolder,
                            filename=filename)


def _GetNodeName(node: WorkflowNode) -> str | None:
  return node.properties.get('Node name for S&R', None)


def _FindNodeByID(*, workflow: Workflow,
                  node_id: str) -> Tuple[str, str | None, WorkflowNode]:
  with WatchVar(node_id=node_id):
    if not isinstance(node_id, str):
      raise ValueError(
          f'Invalid node_id: {node_id} (expected str) got {type(node_id)}')
    for node in workflow.nodes:
      if str(node.id) == node_id:
        return node_id, _GetNodeName(node), node
    raise ValueError(f'Node with id {json.dumps(node_id)} not found')


def _FindNodeByName(*, workflow: Workflow,
                    name: str) -> Tuple[str, str, WorkflowNode]:
  for node in workflow.nodes:
    if _GetNodeName(node) == name:
      return str(node.id), name, node
  raise ValueError(f'Node with name {json.dumps(name)} not found')


def _FindNode(
    *, workflow: Workflow,
    node_id_or_name: int | str) -> Tuple[str, str | None, WorkflowNode]:
  with WatchVar(node_id_or_name=node_id_or_name):
    if isinstance(node_id_or_name, int):
      return _FindNodeByID(workflow=workflow, node_id=str(node_id_or_name))
    if not isinstance(node_id_or_name, str):
      raise ValueError(
          f'Invalid node: {node_id_or_name} (expected int or str) got {type(node_id_or_name)}'
      )
    try:
      return _FindNodeByName(workflow=workflow, name=node_id_or_name)
    except ValueError:
      return _FindNodeByID(workflow=workflow, node_id=str(node_id_or_name))


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


class RemoteFileAPIBase(ABC):

  @abstractmethod
  async def UploadFile(self, *, trusted_src_path: Path,
                       untrusted_dst_io_spec: IOSpec) -> str:
    raise NotImplementedError()

  @abstractmethod
  async def CopyFile(self, *, untrusted_src_io_spec: IOSpec,
                     untrusted_dst_io_spec: IOSpec) -> str:
    raise NotImplementedError()

  @abstractmethod
  async def DownloadFile(self, *, untrusted_src_io_spec: IOSpec,
                         trusted_dst_path: Path) -> None:
    raise NotImplementedError()


def _RenameURLPath(url: str, counter: int) -> str:
  url_pr = urlparse(url)
  path = PurePath(url_pr.path)
  path = path.with_stem(f'{path.stem}_{counter}')
  url_pr = url_pr._replace(path=str(path))
  return url_pr.geturl()


class FSSpecRemoteFileAPI(RemoteFileAPIBase):
  Mode = Literal['r', 'w', 'rw']

  class Overwrite(enum.StrEnum):
    FAIL = enum.auto()
    OVERWRITE = enum.auto()
    RENAME = enum.auto()

  def __init__(self, *, overwrite: Overwrite):
    self._overwrite = overwrite
    self._fs: Dict[FSSpecRemoteFileAPI.Mode,
                   Dict[str, fsspec.spec.AbstractFileSystem]]
    self._fs = {
        'r': {},
        'w': {},
        'rw': {},
    }

  def AddFS(self, url_prefix: str, fs: fsspec.spec.AbstractFileSystem,
            mode: Mode):

    self._fs[mode][url_prefix] = fs
    if mode == 'rw':
      self._fs['r'][url_prefix] = fs
      self._fs['w'][url_prefix] = fs

  async def UploadFile(self, *, trusted_src_path: Path,
                       untrusted_dst_io_spec: IOSpec) -> str:
    trusted_src_path = await trusted_src_path.absolute()
    trusted_src_io_spec = IOSpec(io_url=trusted_src_path.as_uri())
    src_fs = fsspec.filesystem('file')

    dst_fs = self._GetFS(untrusted_dst_io_spec.io_url, 'w')
    trusted_dst_io_spec = untrusted_dst_io_spec

    return await asyncio.to_thread(self._CopyFileSync,
                                   src_fs=src_fs,
                                   trusted_src_io_spec=trusted_src_io_spec,
                                   dst_fs=dst_fs,
                                   trusted_dst_io_spec=trusted_dst_io_spec)

  async def DownloadFile(self, *, untrusted_src_io_spec: IOSpec,
                         trusted_dst_path: Path) -> None:
    with WatchVar(untrusted_src_io_spec=untrusted_src_io_spec,
                  trusted_dst_path=trusted_dst_path):
      src_fs = self._GetFS(untrusted_src_io_spec.io_url, 'r')
      trusted_src_io_spec = untrusted_src_io_spec

      dst_fs = fsspec.filesystem('file')
      trusted_dst_io_spec = IOSpec(io_url=trusted_dst_path.as_uri())

      await asyncio.to_thread(self._CopyFileSync,
                              src_fs=src_fs,
                              trusted_src_io_spec=trusted_src_io_spec,
                              dst_fs=dst_fs,
                              trusted_dst_io_spec=trusted_dst_io_spec)

  async def CopyFile(self, *, untrusted_src_io_spec: IOSpec,
                     untrusted_dst_io_spec: IOSpec) -> str:
    src_fs = self._GetFS(untrusted_src_io_spec.io_url, 'r')
    trusted_src_io_spec = untrusted_src_io_spec
    dst_fs = self._GetFS(untrusted_dst_io_spec.io_url, 'w')
    trusted_dst_io_spec = untrusted_dst_io_spec

    return await asyncio.to_thread(self._CopyFileSync,
                                   src_fs=src_fs,
                                   trusted_src_io_spec=trusted_src_io_spec,
                                   dst_fs=dst_fs,
                                   trusted_dst_io_spec=trusted_dst_io_spec)

  def _FindUnusedFileNameSync(self, fs: fsspec.spec.AbstractFileSystem,
                              url: str) -> str:
    counter = 1
    while True:
      renamed_url = _RenameURLPath(url, counter)
      if not fs.exists(renamed_url):
        return renamed_url
      counter += 1

  def _GetFinalDstURLSync(self, dst_fs: fsspec.spec.AbstractFileSystem,
                          trusted_dst_url: str) -> str:

    if self._overwrite == FSSpecRemoteFileAPI.Overwrite.OVERWRITE:
      return trusted_dst_url
    elif self._overwrite == FSSpecRemoteFileAPI.Overwrite.FAIL:
      if dst_fs.exists(trusted_dst_url):
        raise FileExistsError(f'{trusted_dst_url} already exists')
      return trusted_dst_url
    elif self._overwrite == FSSpecRemoteFileAPI.Overwrite.RENAME:
      if dst_fs.exists(trusted_dst_url):
        return self._FindUnusedFileNameSync(dst_fs, trusted_dst_url)
      return trusted_dst_url
    else:
      raise ValueError(f'Unsupported overwrite: {self._overwrite}')

  def _CopyFileSync(self, *, src_fs: fsspec.spec.AbstractFileSystem,
                    trusted_src_io_spec: IOSpec,
                    dst_fs: fsspec.spec.AbstractFileSystem,
                    trusted_dst_io_spec: IOSpec) -> str:

    final_dst_url = self._GetFinalDstURLSync(dst_fs, trusted_dst_io_spec.io_url)

    if src_fs == dst_fs:
      try:
        # comfyfs will fail this with NotImplementedError.
        src_fs.copy(trusted_src_io_spec, final_dst_url)
        return final_dst_url
      except NotImplementedError:
        pass

    with src_fs.open(trusted_src_io_spec.io_url, 'rb',
                     **trusted_src_io_spec.kwargs) as src:
      with dst_fs.open(final_dst_url, 'wb',
                       **trusted_dst_io_spec.kwargs) as dst:
        for chunk in iter(lambda: src.read(4096), b''):
          dst.write(chunk)  # type: ignore
      if isinstance(dst, _Writable):
        return dst.renamed
    return final_dst_url

  def _GetFS(self, uri: str, mode: Mode) -> fsspec.spec.AbstractFileSystem:
    for url_prefix, fs in self._fs[mode].items():
      if uri.startswith(url_prefix):
        return fs
    raise ValueError(
        f'No matching FS for {json.dumps(uri)}, self._fs: {list(self._fs)}')


async def _DownloadURLToB64(*, remote: RemoteFileAPIBase, io_spec: IOSpec,
                            tmp_dir_path: Path) -> str:
  with WatchVar(io_spec=io_spec, tmp_dir_path=tmp_dir_path):
    filename = PurePath(urlparse(io_spec.io_url).path).name
    async with aiofiles.tempfile.TemporaryDirectory(
        dir=tmp_dir_path) as tmp_child_dir_path:

      tmp_path: Path = Path(str(tmp_child_dir_path)) / slugify(filename)
      tmp_path = await tmp_path.absolute()
      await remote.DownloadFile(untrusted_src_io_spec=io_spec,
                                trusted_dst_path=tmp_path)
      bytes_ = await tmp_path.read_bytes()
      return base64.b64encode(bytes_).decode('utf-8')


async def _UploadDataURI(*, remote: RemoteFileAPIBase, src_uri: str,
                         untrusted_dst_io_spec: IOSpec,
                         tmp_dir_path: Path) -> str:
  data_uri = DataURI(src_uri)
  filename = PurePath(urlparse(untrusted_dst_io_spec.io_url).path).name
  async with aiofiles.tempfile.TemporaryDirectory(
      dir=tmp_dir_path) as tmp_child_dir_path:

    tmp_path: Path = Path(str(tmp_child_dir_path)) / slugify(filename)
    tmp_path = await tmp_path.absolute()

    await tmp_path.write_bytes(data_uri.data)
    return await remote.UploadFile(trusted_src_path=tmp_path,
                                   untrusted_dst_io_spec=untrusted_dst_io_spec)


class ComfyRemoteInfo(BaseModel):
  comfy_api_url: str
  # {prefix => URL}
  upload: Dict[str, IOSpec]
  download: Dict[str, IOSpec]
  logs: IOSpec | None


def _GetRemoteIOSpec(*, path: PurePath, prefix2iospec: Dict[str,
                                                            IOSpec]) -> IOSpec:
  if path.is_absolute():
    raise ValueError(f'path must not be absolute: {path}')
  path_str = str(path)
  for prefix, io_spec in prefix2iospec.items():
    if path_str.startswith(prefix):
      path_io_url: str = io_spec.io_url + path_str[len(prefix):]
      return io_spec.model_copy(deep=True, update={'io_url': path_io_url})
  raise ValueError(
      f'No matching prefix for {json.dumps(path)}, prefix2url: {prefix2iospec}')


class ProvisioningBundle(BaseModel):
  files: Dict[str, IOSpec]
  archives: Dict[str, IOSpec]
  custom_nodes: Dict[str, IOSpec]


class InputPPKind(enum.Enum):
  VALUE = enum.auto()
  """Anything JSON Serializable. Just gets passed through."""
  FILE = enum.auto()
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


def _ParseUserIOSpec(user_input_value: JSONSerializable) -> UserIOSpec:
  io_spec_validator = pydantic.TypeAdapter[UserIOSpec](UserIOSpec)
  if not isinstance(user_input_value, str):
    return io_spec_validator.validate_python(user_input_value)
  try:
    return io_spec_validator.validate_json(user_input_value)
  except pydantic.ValidationError:
    return UserIOSpec(root=user_input_value)


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


class OutputPPKind(enum.StrEnum):
  NODE_VALUE = enum.auto()
  """Anything JSON Serializable. Just gets passed through as an output value."""
  FILE = enum.auto()
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
  pp_spec: FileUploadMapSpec | None = Field(
      None, description='If pp==FILE, then this is required.', alias='spec')

  @model_validator(mode='after')
  def check_exclusivity(self):
    if (self.pp == InputPPKind.FILE) != (self.pp_spec is not None):
      raise ValueError(
          'If pp!=VALUE, then pp_spec is required. Otherwise, it must be None.')
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
  pp_spec: FileDownloadMapSpec | None = Field(
      ..., description='If pp!=VALUE, then this is required.', alias='spec')
  user_json_spec: Literal['NO_PUBLIC_INPUT'] | Literal['ANY'] | Literal[
      'OPT_ANY'] | dict = Field(
          ..., description='JSON spec to verify the user input.')
  user_value: JSONSerializable = Field(
      ..., description='The default input value to use.')

  @model_validator(mode='after')
  def check_exclusivity(self):
    if (self.pp == OutputPPKind.FILE) != (self.pp_spec is not None):
      raise ValueError(
          'If pp!=VALUE, then pp_spec is required. Otherwise, it must be None.')
    return self


class WorkflowTemplateBundle(BaseModel):
  workflow_template: Workflow
  api_workflow_template: APIWorkflow
  important: List[APINodeID]
  object_info: APIObjectInfo
  input_mappings: List[InputMapping]
  output_mappings: List[OutputMapping]


class ProvisionerBase(ABC):

  class ProvisionReq(BaseModel):
    id: str
    bundle: ProvisioningBundle
    keepalive: float

  class ProvisionRes(BaseModel):
    id: str
    comfy_info: ComfyRemoteInfo

  class TouchReq(BaseModel):
    id: str
    keepalive: float

  class TouchRes(BaseModel):
    success: bool
    message: str

  @abstractmethod
  async def Provision(self, req: ProvisionReq) -> ProvisionRes:
    raise NotImplementedError()

  @abstractmethod
  async def Touch(self, req: TouchReq) -> TouchRes:
    raise NotImplementedError()


class DumbProvisioner(ProvisionerBase):

  def __init__(self, *, comfy_remote: ComfyRemoteInfo):
    self._comfy_remote = comfy_remote

  async def Provision(
      self, req: ProvisionerBase.ProvisionReq) -> ProvisionerBase.ProvisionRes:
    return ProvisionerBase.ProvisionRes(id=req.id,
                                        comfy_info=self._comfy_remote)

  async def Touch(self,
                  req: ProvisionerBase.TouchReq) -> ProvisionerBase.TouchRes:
    return ProvisionerBase.TouchRes(success=True, message='')


class ServerBase(ABC):

  class UploadWorkflowReq(BaseModel):
    workflow_id: str
    template_bundle: WorkflowTemplateBundle
    provisioning: ProvisioningBundle
    keepalive: float

  class UploadWorkflowRes(BaseModel):
    pass

  @abstractmethod
  async def UploadWorkflow(self, req: UploadWorkflowReq) -> UploadWorkflowRes:
    raise NotImplementedError()

  class ExecuteReq(BaseModel):
    job_id: str
    workflow_id: str
    user_input_values: Dict[str, JSONSerializable]

  class ExecuteRes(BaseModel):
    job_id: str
    history: APIHistoryEntry
    mapped_outputs: Dict[str, JSONSerializable]

  @abstractmethod
  async def Execute(self, req: ExecuteReq) -> ExecuteRes:
    raise NotImplementedError()

  class TouchReq(BaseModel):
    job_id: str
    keepalive: float

  class TouchRes(BaseModel):
    success: bool
    message: str

  @abstractmethod
  async def Touch(self, req: TouchReq) -> TouchRes:
    raise NotImplementedError()


def _GetTripletFileIOSpec(*, triplet: ComfyUIPathTriplet,
                          remote_info: ComfyRemoteInfo,
                          mode: Literal['upload', 'download']) -> IOSpec:
  remote_path = triplet.ToLocalPathStr(include_folder_type=True)
  if mode == 'download':
    download_io_spec = _GetRemoteIOSpec(path=PurePath(remote_path),
                                        prefix2iospec=remote_info.download)
    return download_io_spec
  elif mode == 'upload':
    upload_io_spec = _GetRemoteIOSpec(path=PurePath(remote_path),
                                      prefix2iospec=remote_info.upload)
    return upload_io_spec
  else:
    raise ValueError(f'Unsupported mode: {json.dumps(mode)}')


class PreProcessorBase(ABC):

  def Validate(self, user_input_value: JSONSerializable):
    if not isinstance(user_input_value, JSON_SERIALIZABLE_TYPES):
      raise ValueError('Expected user_input_value to be one of '
                       f'{JSON_SERIALIZABLE_TYPES}, '
                       f'got {type(user_input_value)}')
    if not _IsJSONSerializable(user_input_value):
      raise ValueError('Expected IsJSONSerializable(user_input_value)==True')

  @abstractmethod
  async def ProcessInput(
      self, comfy_info: ComfyRemoteInfo, mapping: InputMapping,
      user_input_value: JSONSerializable) -> JSONSerializable:
    raise NotImplementedError()


class ValuePreProcessor(PreProcessorBase):

  async def ProcessInput(
      self, comfy_info: ComfyRemoteInfo, mapping: InputMapping,
      user_input_value: JSONSerializable) -> JSONSerializable:
    self.Validate(user_input_value)

    if not isinstance(user_input_value, JSON_SERIALIZABLE_TYPES):
      raise ValueError('Expected user_input_value to be one of '
                       f'{JSON_SERIALIZABLE_TYPES}, because '
                       f'InputMapping.pp={json.dumps(mapping.pp)}, '
                       f'got {type(user_input_value)}')
    if not _IsJSONSerializable(user_input_value):
      raise ValueError(
          'Expected user_input_value to be JSONSerializable, because '
          f'InputMapping.pp={json.dumps(mapping.pp)}, '
          f'got {type(user_input_value)}')
    return user_input_value


class FilePreProcessor(PreProcessorBase):

  def __init__(self, *, remote: RemoteFileAPIBase, tmp_dir_path: Path) -> None:
    super().__init__()
    self._remote = remote
    self._tmp_dir_path = tmp_dir_path

  async def ProcessInput(
      self, comfy_info: ComfyRemoteInfo, mapping: InputMapping,
      user_input_value: JSONSerializable) -> JSONSerializable:
    self.Validate(user_input_value)

    if mapping.pp_spec is None:
      raise ValueError('mapping.spec is None for mapping.pp=FILE')
    file_mapping_spec: FileUploadMapSpec = mapping.pp_spec

    src_io_spec = _ParseUserIOSpec(user_input_value).ToIOSpec()

    dst_io_spec: IOSpec
    upload_to_path: PurePath
    if isinstance(file_mapping_spec.upload_to, ComfyUIPathTriplet):
      upload_to_path = PurePath(
          file_mapping_spec.upload_to.ToLocalPathStr(include_folder_type=True))
      dst_io_spec = _GetTripletFileIOSpec(triplet=file_mapping_spec.upload_to,
                                          remote_info=comfy_info,
                                          mode='upload')
    else:
      upload_to_path = PurePath(file_mapping_spec.upload_to)
      dst_io_spec = _GetRemoteIOSpec(path=upload_to_path,
                                     prefix2iospec=comfy_info.upload)

    remote_url: str
    if src_io_spec.io_url.startswith('data:'):
      remote_url = await _UploadDataURI(remote=self._remote,
                                        src_uri=src_io_spec.io_url,
                                        untrusted_dst_io_spec=dst_io_spec,
                                        tmp_dir_path=self._tmp_dir_path)
    else:
      remote_url = await self._remote.CopyFile(
          untrusted_src_io_spec=src_io_spec, untrusted_dst_io_spec=dst_io_spec)

    remote_filename = PurePath(urlparse(remote_url).path).name
    uploaded_to_path = upload_to_path.with_name(remote_filename)

    if file_mapping_spec.node_mode == 'FILEPATH':
      # TODO: Might need to do something different here for windows.
      uploaded_to_path_str = uploaded_to_path.as_posix()
      return _DiscardPrefix(uploaded_to_path_str,
                            file_mapping_spec.discard_prefix)
    elif file_mapping_spec.node_mode == 'TRIPLET':
      dst_triplet: ComfyUIPathTriplet = _PathToTriplet(upload_to_path)
      return dst_triplet.model_dump(mode='json', round_trip=True)
    else:
      raise ValueError(f'Unsupported node_mode: {file_mapping_spec.node_mode}')


class PostProcessorBase(ABC):

  @abstractmethod
  async def ProcessOutput(
      self, comfy_info: ComfyRemoteInfo, mapping: OutputMapping,
      user_input_value: JSONSerializable,
      node_output_value: JSONSerializable) -> JSONSerializable:
    raise NotImplementedError()

  def Validate(self, *, user_input_value: JSONSerializable,
               node_output_value: JSONSerializable):
    if not isinstance(user_input_value, JSON_SERIALIZABLE_TYPES):
      raise ValueError('Expected user_input_value to be one of '
                       f'{JSON_SERIALIZABLE_TYPES}, '
                       f'got {type(user_input_value)}')
    if not _IsJSONSerializable(user_input_value):
      raise ValueError('Expected IsJSONSerializable(user_output_value)==True')
    if not isinstance(node_output_value, JSON_SERIALIZABLE_TYPES):
      raise ValueError('Expected node_output_value to be one of '
                       f'{JSON_SERIALIZABLE_TYPES}, '
                       f'got {type(node_output_value)}')
    if not _IsJSONSerializable(node_output_value):
      raise ValueError('Expected IsJSONSerializable(node_output_value)==True')


class ValuePostProcessor(PostProcessorBase):

  async def ProcessOutput(
      self, comfy_info: ComfyRemoteInfo, mapping: OutputMapping,
      user_input_value: JSONSerializable,
      node_output_value: JSONSerializable) -> JSONSerializable:
    self.Validate(user_input_value=user_input_value,
                  node_output_value=node_output_value)
    return user_input_value


class FilePostProcessor(PostProcessorBase):

  def __init__(self, *, remote: RemoteFileAPIBase, tmp_dir_path: Path):
    self._remote = remote
    self._tmp_dir_path = tmp_dir_path

  async def ProcessOutput(
      self, comfy_info: ComfyRemoteInfo, mapping: OutputMapping,
      user_input_value: JSONSerializable,
      node_output_value: JSONSerializable) -> JSONSerializable:
    self.Validate(user_input_value=user_input_value,
                  node_output_value=node_output_value)
    triplet = ComfyUIPathTriplet.model_validate(node_output_value)
    download_io_spec = _GetTripletFileIOSpec(triplet=triplet,
                                             remote_info=comfy_info,
                                             mode='download')

    dst_user_io_spec = _ParseUserIOSpec(user_input_value)
    if dst_user_io_spec.root == 'base64':
      return await _DownloadURLToB64(remote=self._remote,
                                     io_spec=download_io_spec,
                                     tmp_dir_path=self._tmp_dir_path)
    dst_io_spec = dst_user_io_spec.ToIOSpec()
    return await self._remote.CopyFile(untrusted_src_io_spec=download_io_spec,
                                       untrusted_dst_io_spec=dst_io_spec)


def _GetInputValue(mapping: InputMapping | OutputMapping,
                   user_input_values: Dict[str, JSONSerializable],
                   error_context: _ErrorContext) -> JSONSerializable:
  if mapping.user_json_spec == 'NO_PUBLIC_INPUT':
    if mapping.name in user_input_values:
      raise ValueError(
          f'input_mapping.name={json.dumps(mapping.name)} provided by the user, but input_mapping.user_json_spec={json.dumps(mapping.user_json_spec)}'
      )
    return mapping.user_value
  elif mapping.user_json_spec == 'OPT_ANY':
    return user_input_values.get(mapping.name, mapping.user_value)
  else:
    if mapping.name not in user_input_values:
      raise ValueError(
          f'input_mapping.name={json.dumps(mapping.name)}'
          f' provided by the user not in user_input_values {json.dumps(list(user_input_values.keys()))}'
          f'\n\n{textwrap.indent(YamlDump(error_context.Dump()), "  ")}')
    value = user_input_values[mapping.name]
    if mapping.user_json_spec == 'ANY':
      return value
    jsonschema.validate(
        instance=value,
        schema=mapping.user_json_spec,
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)


class Server(ServerBase):
  UploadWorkflowReq = ServerBase.UploadWorkflowReq
  UploadWorkflowRes = ServerBase.UploadWorkflowRes
  ExecuteReq = ServerBase.ExecuteReq
  ExecuteRes = ServerBase.ExecuteRes

  TouchReq = ServerBase.TouchReq
  TouchRes = ServerBase.TouchRes

  def __init__(self, *, provisioner: ProvisionerBase, remote: RemoteFileAPIBase,
               tmp_dir_path: Path, debug_path: Path,
               debug_save_all: bool) -> None:
    self._provisioner = provisioner
    self._remote = remote
    self._tmp_dir_path = tmp_dir_path
    self._input_processors: Dict[InputPPKind, PreProcessorBase] = {}
    self._post_processors: Dict[OutputPPKind, PostProcessorBase] = {}
    self._debug_path: Path = debug_path
    self._debug_save_all = debug_save_all

    self._input_processors[InputPPKind.VALUE] = ValuePreProcessor()
    self._input_processors[InputPPKind.FILE] = FilePreProcessor(
        remote=remote, tmp_dir_path=tmp_dir_path)
    self._post_processors[OutputPPKind.NODE_VALUE] = ValuePostProcessor()
    self._post_processors[OutputPPKind.FILE] = FilePostProcessor(
        remote=self._remote, tmp_dir_path=self._tmp_dir_path)

    self._workflows: Dict[str, ServerBase.UploadWorkflowReq] = {}

  async def UploadWorkflow(
      self, req: ServerBase.UploadWorkflowReq) -> ServerBase.UploadWorkflowRes:
    self._workflows[req.workflow_id] = req
    return ServerBase.UploadWorkflowRes()

  async def _KeepAlive(self, job_id: str, keepalive: float) -> None:
    try:
      while True:
        await asyncio.sleep(keepalive / 2)
        await self._provisioner.Touch(
            ProvisionerBase.TouchReq(id=job_id, keepalive=keepalive))
    except Exception:
      logger.exception('KeepAlive failed',
                       exc_info=True,
                       stack_info=True,
                       extra={
                           'job_id': job_id,
                           'keepalive': keepalive
                       })

  async def _UploadValue(self, comfy_info: ComfyRemoteInfo,
                         workflow_template: Workflow,
                         prepared_api_workflow: APIWorkflow,
                         input_mapping: InputMapping,
                         user_input_value: JSONSerializable) -> None:
    node_id: APINodeID
    node_id, _, _ = _FindNode(workflow=workflow_template,
                              node_id_or_name=input_mapping.node)
    pp: InputPPKind = input_mapping.pp
    if node_id not in prepared_api_workflow.root:
      raise ValueError(f'{node_id} not in prepared_workflow.root')
    node: APIWorkflowNodeInfo = prepared_api_workflow.root[node_id]
    if pp not in self._input_processors:
      raise ValueError(f'pp={pp} not in self._input_processors')
    processor = self._input_processors[pp]
    user_input_value_pp: JSONSerializable
    user_input_value_pp = await processor.ProcessInput(
        comfy_info=comfy_info,
        mapping=input_mapping,
        user_input_value=user_input_value)
    # TODO: Ensure that this field path exists.
    pydash.set_(node.inputs, input_mapping.field, user_input_value_pp)

  async def _UploadValues(self, comfy_info: ComfyRemoteInfo,
                          workflow_template: Workflow,
                          prepared_api_workflow: APIWorkflow,
                          input_mappings: List[InputMapping],
                          user_input_values: Dict[str, JSONSerializable],
                          error_context: _ErrorContext) -> None:

    error_context['input_mappings'] = [
        m.model_dump(mode='json', round_trip=True) for m in input_mappings
    ]
    error_context['user_input_values'] = user_input_values
    for input_mapping in input_mappings:
      error_context['input_mapping'] = input_mapping.model_dump(mode='json',
                                                                round_trip=True)
      try:

        user_input_value = _GetInputValue(input_mapping, user_input_values,
                                          error_context)
        error_context['user_input_value'] = user_input_value

        await self._UploadValue(comfy_info=comfy_info,
                                workflow_template=workflow_template,
                                prepared_api_workflow=prepared_api_workflow,
                                input_mapping=input_mapping,
                                user_input_value=user_input_value)
      except Exception as e:
        logger.exception('UploadValue failed',
                         exc_info=True,
                         stack_info=True,
                         extra=error_context.Dump())
        raise ValueError(
            f'Failed to upload {json.dumps(input_mapping.name)}'
            f'\n\n{textwrap.indent(YamlDump(error_context.Dump()), "  ")}'
        ) from e

  async def _DownloadValue(self, comfy_info: ComfyRemoteInfo,
                           workflow_template: Workflow,
                           prepared_api_workflow: APIWorkflow,
                           history: APIHistoryEntry,
                           output_mapping: OutputMapping,
                           user_input_values: Dict[str, JSONSerializable],
                           error_context: _ErrorContext) -> JSONSerializable:
    # TODO: Wrap this in a try-catch and throw a specific error for this input.
    node_id: APINodeID
    node_id, _, _ = _FindNode(workflow=workflow_template,
                              node_id_or_name=output_mapping.node)
    pp: OutputPPKind = output_mapping.pp
    if history.outputs is None or node_id not in history.outputs:
      raise ValueError(f'{node_id} not in history.outputs')

    node_outputs: APIOutputUI = history.outputs[node_id]
    node_output_value: JSONSerializable
    node_output_value_any: Any = pydash.get(node_outputs.root,
                                            output_mapping.field)

    if not isinstance(node_output_value_any, JSON_SERIALIZABLE_TYPES):
      raise ValueError(
          f'Expected node_output_value to be one of {JSON_SERIALIZABLE_TYPES}, got {type(node_output_value_any)}'
      )
    if not _IsJSONSerializable(node_output_value_any):
      raise ValueError(
          f'Expected node_output_value to be JSONSerializable, got {type(node_output_value_any)}'
      )
    node_output_value = node_output_value_any
    if pp not in self._post_processors:
      raise ValueError(f'pp={pp} not in self._output_processors')
    processor = self._post_processors[pp]
    user_input_value: JSONSerializable
    user_input_value = _GetInputValue(output_mapping,
                                      user_input_values=user_input_values,
                                      error_context=error_context.Copy())
    return await processor.ProcessOutput(comfy_info=comfy_info,
                                         mapping=output_mapping,
                                         user_input_value=user_input_value,
                                         node_output_value=node_output_value)

  async def _DownloadValues(
      self, comfy_info: ComfyRemoteInfo, workflow_template: Workflow,
      output_mappings: List[OutputMapping], prepared_api_workflow: APIWorkflow,
      history: APIHistoryEntry, user_input_values: Dict[str, JSONSerializable],
      error_context: _ErrorContext) -> Dict[str, JSONSerializable]:
    user_output_values: Dict[str, JSONSerializable] = {}
    error_context['user_output_values'] = user_output_values

    output_mapping: OutputMapping
    for output_mapping in output_mappings:
      error_context['user_output_name'] = output_mapping.name
      try:
        if output_mapping.name in user_output_values:
          raise ValueError(
              f'output_mapping.name={json.dumps(output_mapping.name)} already in user_output_values'
          )
        error_context['output_mapping'] = output_mapping.model_dump(
            mode='json', round_trip=True)
        value = await self._DownloadValue(
            comfy_info=comfy_info,
            workflow_template=workflow_template,
            prepared_api_workflow=prepared_api_workflow,
            history=history,
            output_mapping=output_mapping,
            user_input_values=user_input_values,
            error_context=error_context.Copy())
        user_output_values[output_mapping.name] = value
        logging.info('DownloadValue succeeded', extra=error_context.Dump())
      except Exception as e:
        logger.exception('DownloadValue failed',
                         exc_info=True,
                         stack_info=True,
                         extra=error_context.Dump())
        raise ValueError(
            f'Failed to download {json.dumps(output_mapping.name)}'
            f'\n\n{textwrap.indent(YamlDump(error_context.Dump()), "  ")})'
        ) from e

    return user_output_values

  async def _Execute(self, req: ExecuteReq) -> ExecuteRes:
    if req.workflow_id not in self._workflows:
      raise ValueError(f'Workflow not found: {req.workflow_id}')
    upload_req: ServerBase.UploadWorkflowReq = self._workflows[req.workflow_id]
    provisioning: ProvisioningBundle = upload_req.provisioning

    error_context = _ErrorContext(debug_path=self._debug_path, key=req.job_id)
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True))
    error_context['upload_req'] = await error_context.LargeToFile(
        'upload_req', upload_req.model_dump(mode='json', round_trip=True))
    error_context['provisioning'] = provisioning.model_dump(mode='json',
                                                            round_trip=True)
    error_context['keepalive'] = upload_req.keepalive
    error_context['user_input_values'] = await error_context.LargeToFile(
        'user_input_values', req.user_input_values)

    logger.info('Manager.Execute() was called', extra=error_context.Dump())

    ############################################################################
    if slugify(req.job_id) != req.job_id:
      raise ValueError(f'job_id must be slugified: {req.job_id}')

    job_debug_path = self._debug_path / f'{slugify(datetime.now().isoformat())}_{req.job_id}'
    await job_debug_path.mkdir(parents=True, exist_ok=True)
    ############################################################################
    ProvisionReq = ProvisionerBase.ProvisionReq
    provisioned: ProvisionerBase.ProvisionRes
    provisioned = await self._provisioner.Provision(
        ProvisionReq(id=req.job_id,
                     bundle=provisioning,
                     keepalive=upload_req.keepalive))
    error_context['provisioned'] = provisioned.model_dump(mode='json',
                                                          round_trip=True)
    logger.info('Successfully provisioned a ComfyUI instance',
                extra=error_context.Dump())
    ############################################################################
    keepalive_task = asyncio.create_task(
        self._KeepAlive(req.job_id, upload_req.keepalive))
    ############################################################################

    try:
      async with ComfyAPIClient(
          comfy_api_url=provisioned.comfy_info.comfy_api_url) as client:
        async with ComfyCatapult(
            comfy_client=client,
            debug_path=self._debug_path,
            debug_save_all=self._debug_save_all) as catapult:
          api_workflow_template = upload_req.template_bundle.api_workflow_template
          prepared_api_workflow = api_workflow_template.model_copy(deep=True)

          ######################################################################
          await self._UploadValues(
              comfy_info=provisioned.comfy_info,
              workflow_template=upload_req.template_bundle.workflow_template,
              prepared_api_workflow=prepared_api_workflow,
              input_mappings=upload_req.template_bundle.input_mappings,
              user_input_values=req.user_input_values,
              error_context=error_context.Copy())
          error_context['prepared_workflow'] = await error_context.LargeToFile(
              'prepared_workflow',
              prepared_api_workflow.model_dump(mode='json', round_trip=True))
          logger.info('Uploaded all user input values',
                      extra=error_context.Dump())
          ######################################################################

          history_dict = await catapult.Catapult(
              job_id=req.job_id,
              prepared_workflow=prepared_api_workflow.model_dump(
                  mode='json', round_trip=True),
              important=upload_req.template_bundle.important,
              job_debug_path=job_debug_path)
          error_context['history_dict'] = await error_context.LargeToFile(
              'history_dict', history_dict)
          history: APIHistoryEntry = APIHistoryEntry.model_validate(
              history_dict)
          error_context['history'] = await error_context.LargeToFile(
              'history', history.model_dump(mode='json', round_trip=True))
          logger.info('Catapulted the job', extra=error_context.Dump())
          ######################################################################
          mapped_outputs: Dict[str, JSONSerializable]
          mapped_outputs = await self._DownloadValues(
              comfy_info=provisioned.comfy_info,
              workflow_template=upload_req.template_bundle.workflow_template,
              output_mappings=upload_req.template_bundle.output_mappings,
              prepared_api_workflow=prepared_api_workflow,
              history=history,
              user_input_values=req.user_input_values,
              error_context=error_context.Copy())
          error_context['mapped_outputs'] = await error_context.LargeToFile(
              'mapped_outputs', mapped_outputs)
          logger.info('Downloaded all user output values',
                      extra=error_context.Dump())
          ######################################################################
          return self.ExecuteRes(job_id=req.job_id,
                                 history=history,
                                 mapped_outputs=mapped_outputs)
    finally:
      keepalive_task.cancel()

  async def Execute(self, req: ExecuteReq) -> ExecuteRes:
    return await self._Execute(req)

  async def Touch(self, req: TouchReq) -> TouchRes:
    res: ProvisionerBase.TouchRes
    res = await self._provisioner.Touch(
        ProvisionerBase.TouchReq(id=req.job_id, keepalive=req.keepalive))
    return Server.TouchRes(success=res.success, message=res.message)
