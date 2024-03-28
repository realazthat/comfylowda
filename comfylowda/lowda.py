# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import asyncio
import base64
import enum
import json
import logging
import textwrap
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import PurePath
from typing import Any, Dict, List, Literal, Tuple
from urllib.parse import urlparse

import aiofiles
import fsspec
import jsonschema
import pydantic
import pydash
from anyio import Path
from comfy_catapult.api_client import ComfyAPIClient
from comfy_catapult.catapult import ComfyCatapult
from comfy_catapult.comfy_schema import (APIHistoryEntry, APINodeID,
                                         APIOutputUI, APIWorkflow,
                                         APIWorkflowNodeInfo, ComfyFolderType,
                                         ComfyUIPathTriplet)
from comfy_catapult.comfy_utils import WatchVar
from datauri import DataURI
from pydantic import BaseModel
from slugify import slugify

from comfylowda import lowda_types
from comfylowda.validation import _CheckModelRoundTrip

from .comfy_schema import Workflow, WorkflowNode
from .comfyfs import _Writable
from .error_utils import _ErrorContext, _YamlDump
from .lowda_types import (JSON_SERIALIZABLE_TYPES, ComfyRemoteInfo,
                          FileUploadMapSpec, InputMapping, InputPPKind, IOSpec,
                          IsJSONSerializable, JSONSerializable, OutputMapping,
                          OutputPPKind, ProvisioningSpec, UploadWorkflowError,
                          UserIOSpec)

logger = logging.getLogger(__name__)


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


class NoMatchingFS(Exception):
  pass


class FSSpecRemoteFileAPI(RemoteFileAPIBase):
  Mode = Literal['r', 'w', 'rw']

  class Overwrite(str, enum.Enum):
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

  def _GetRegisteredPrefixes(self, mode: Mode) -> List[str]:
    return list(self._fs[mode].keys())

  def _GetFS(self, url: str, mode: Mode) -> fsspec.spec.AbstractFileSystem:
    for url_prefix, fs in self._fs[mode].items():
      if url.startswith(url_prefix):
        return fs

    prefixes = self._GetRegisteredPrefixes(mode)
    raise NoMatchingFS(f'No matching FS for {url}, in mode {mode}, registered '
                       f'prefixes for this mode: {prefixes}')


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


async def _UploadDataURI(*, remote: RemoteFileAPIBase, src_data_uri: str,
                         untrusted_dst_io_spec: IOSpec,
                         tmp_dir_path: Path) -> str:
  data_uri = DataURI(src_data_uri)
  filename = PurePath(urlparse(untrusted_dst_io_spec.io_url).path).name
  async with aiofiles.tempfile.TemporaryDirectory(
      dir=tmp_dir_path) as tmp_child_dir_path:

    tmp_path: Path = Path(str(tmp_child_dir_path)) / slugify(filename)
    tmp_path = await tmp_path.absolute()

    await tmp_path.write_bytes(data_uri.data)
    return await remote.UploadFile(trusted_src_path=tmp_path,
                                   untrusted_dst_io_spec=untrusted_dst_io_spec)


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


def _ParseUserIOSpec(user_input_value: JSONSerializable) -> UserIOSpec:
  io_spec_validator = pydantic.TypeAdapter[UserIOSpec](UserIOSpec)
  if not isinstance(user_input_value, str):
    return io_spec_validator.validate_python(user_input_value)
  try:
    return io_spec_validator.validate_json(user_input_value)
  except pydantic.ValidationError:
    return UserIOSpec(root=user_input_value)


class ProvisionerBase(ABC):

  class ProvisionReq(BaseModel):
    id: str
    bundle: ProvisioningSpec
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
  UploadWorkflowReq = lowda_types.UploadWorkflowReq
  UploadWorkflowError = lowda_types.UploadWorkflowError
  UploadWorkflowRes = lowda_types.UploadWorkflowRes
  DownloadWorkflowReq = lowda_types.DownloadWorkflowReq
  DownloadWorkflowSuccess = lowda_types.DownloadWorkflowSuccess
  DownloadWorkflowError = lowda_types.DownloadWorkflowError
  DownloadWorkflowRes = lowda_types.DownloadWorkflowRes
  ExecuteReq = lowda_types.ExecuteReq
  ExecuteSuccess = lowda_types.ExecuteSuccess
  ExecuteError = lowda_types.ExecuteError
  ExecuteRes = lowda_types.ExecuteRes
  TouchReq = lowda_types.TouchReq
  TouchRes = lowda_types.TouchRes

  @abstractmethod
  async def UploadWorkflow(self, req: UploadWorkflowReq) -> UploadWorkflowRes:
    raise NotImplementedError()

  @abstractmethod
  async def Execute(self, req: ExecuteReq) -> ExecuteRes:
    raise NotImplementedError()

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
    if not IsJSONSerializable(user_input_value):
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
    if not IsJSONSerializable(user_input_value):
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

    if mapping.spec is None:
      raise ValueError('mapping.spec is None for mapping.pp=FILE')
    file_mapping_spec: FileUploadMapSpec = mapping.spec

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
                                        src_data_uri=src_io_spec.io_url,
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
    if not IsJSONSerializable(user_input_value):
      raise ValueError('Expected IsJSONSerializable(user_output_value)==True')
    if not isinstance(node_output_value, JSON_SERIALIZABLE_TYPES):
      raise ValueError('Expected node_output_value to be one of '
                       f'{JSON_SERIALIZABLE_TYPES}, '
                       f'got {type(node_output_value)}')
    if not IsJSONSerializable(node_output_value):
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
          f'\n\n{textwrap.indent(_YamlDump(error_context.Dump()), "  ")}')
    value = user_input_values[mapping.name]
    if mapping.user_json_spec == 'ANY':
      return value
    jsonschema.validate(
        instance=value,
        schema=mapping.user_json_spec,
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)


class _Error(Exception):

  def __init__(self, *, user_message: str, internal_message: str,
               status_code: int | None, error_name: str,
               io_name: str | None) -> None:
    super().__init__(internal_message)
    self.user_message = user_message
    self.internal_message = internal_message
    self.status_code = status_code
    self.error_name = error_name
    self.io_name = io_name


class Server(ServerBase):
  UploadWorkflowReq = ServerBase.UploadWorkflowReq
  UploadWorkflowRes = ServerBase.UploadWorkflowRes
  DownloadWorkflowReq = ServerBase.DownloadWorkflowReq
  DownloadWorkflowRes = ServerBase.DownloadWorkflowRes
  ExecuteReq = ServerBase.ExecuteReq
  ExecuteRes = ServerBase.ExecuteRes

  TouchReq = ServerBase.TouchReq
  TouchRes = ServerBase.TouchRes

  @classmethod
  async def Create(cls, *, provisioner: ProvisionerBase,
                   remote: RemoteFileAPIBase, tmp_dir_path: Path,
                   debug_path: Path, debug_save_all: bool) -> 'Server':
    return cls(provisioner=provisioner,
               remote=remote,
               tmp_dir_path=tmp_dir_path,
               debug_path=debug_path,
               debug_save_all=debug_save_all,
               _private_please_use_create=cls._PrivatePleaseUseCreate())

  class _PrivatePleaseUseCreate:
    pass

  def __init__(self, *, provisioner: ProvisionerBase, remote: RemoteFileAPIBase,
               tmp_dir_path: Path, debug_path: Path, debug_save_all: bool,
               _private_please_use_create: _PrivatePleaseUseCreate) -> None:
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
    error_context = await _ErrorContext.Create(debug_path=self._debug_path,
                                               key='UploadWorkflow')
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True))
    error_context['workflow_id'] = req.workflow_id
    logger.info('UploadWorkflow', extra={'error_context': error_context.Dump()})
    try:
      self._workflows[req.workflow_id] = req
      return ServerBase.UploadWorkflowRes(error=None)
    except Exception:
      error_id = str(uuid.uuid4())
      status_code = 500
      error_name = 'UploadWorkflowError'
      message = 'Internal Server Error'

      error_context['error_id'] = error_id
      error_context['status_code'] = status_code
      error_context['error_name'] = error_name
      error_context['message'] = message

      logger.exception('UploadWorkflow failed',
                       exc_info=True,
                       stack_info=True,
                       extra={'error_context': error_context.Dump()})
      return ServerBase.UploadWorkflowRes(
          error=UploadWorkflowError(error_id=error_id,
                                    status_code=status_code,
                                    name=error_name,
                                    message=message,
                                    context=error_context.UserDump()))

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
        error_name = 'UploadError'
        status_code = 500
        user_message = 'Internal Server Error'
        internal_message = f'Failed to upload {json.dumps(input_mapping.name)}, e: {e}'
        if isinstance(e, NoMatchingFS):
          error_name = 'NoMatchingFS'
          status_code = 400
          user_message = f'Error with uploading {json.dumps(input_mapping.name)}: {e}'

        raise _Error(user_message=user_message,
                     internal_message=internal_message,
                     status_code=status_code,
                     error_name=error_name,
                     io_name=input_mapping.name) from e

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
    if not IsJSONSerializable(node_output_value_any):
      raise ValueError(
          f'Expected node_output_value to be JSONSerializable, got {type(node_output_value_any)}'
      )
    node_output_value = node_output_value_any
    if pp not in self._post_processors:
      raise ValueError(
          f'pp={repr(pp)} (type(pp)={type(pp)}) not in self._post_processors,'
          f' self._post_processors: {list(self._post_processors.keys())}')
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
        error_name = 'DownloadError'
        status_code = 500
        internal_message = f'Failed to download {json.dumps(output_mapping.name)}, e: {e}'
        user_message = 'Internal Server Error'
        if isinstance(e, NoMatchingFS):
          error_name = 'NoMatchingFS'
          status_code = 400
          user_message = f'Error with downloading {json.dumps(output_mapping.name)}: {e}'
        raise _Error(
            user_message=user_message,
            internal_message=internal_message,
            status_code=status_code,
            error_name=error_name,
            io_name=output_mapping.name,
        ) from e

    return user_output_values

  async def _DownloadWorkflow(
      self, req: DownloadWorkflowReq,
      error_context: _ErrorContext) -> ServerBase.DownloadWorkflowSuccess:
    upload_req: ServerBase.UploadWorkflowReq = self._workflows[req.workflow_id]
    error_context['upload_req'] = await error_context.LargeToFile(
        'upload_req', upload_req.model_dump(mode='json', round_trip=True))
    return ServerBase.DownloadWorkflowSuccess(
        workflow_id=upload_req.workflow_id,
        template_bundle=upload_req.template_bundle,
        prov_spec=upload_req.prov_spec)

  async def DownloadWorkflow(self,
                             req: DownloadWorkflowReq) -> DownloadWorkflowRes:
    error_context = await _ErrorContext.Create(debug_path=self._debug_path,
                                               key='DownloadWorkflow')
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True))
    error_context['req.workflow_id'] = req.workflow_id
    try:
      success = await self._DownloadWorkflow(req,
                                             error_context=error_context.Copy())
      return ServerBase.DownloadWorkflowRes(success=success, error=None)
    except Exception as e:
      error_id = uuid.uuid4().hex
      status_code = 500
      error_name = 'InternalError'
      message = f'An internal error occurred. Please report this error with the error_id: {json.dumps(error_id)}'
      if isinstance(e, KeyError):
        error_name = 'WorkflowNotFound'
        status_code = 404
        message = f'Workflow not found: {req.workflow_id}'
      logger.exception('DownloadWorkflow failed',
                       exc_info=True,
                       stack_info=True,
                       extra=error_context.Dump())
      error = ServerBase.DownloadWorkflowError(error_id=error_id,
                                               status_code=status_code,
                                               name=error_name,
                                               message=message,
                                               context=error_context.UserDump())
      return ServerBase.DownloadWorkflowRes(success=None, error=error)

  async def _Execute(self, req: ExecuteReq,
                     error_context: _ErrorContext) -> ServerBase.ExecuteSuccess:
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True))
    error_context['req.workflow_id'] = req.workflow_id

    _CheckModelRoundTrip(model=req, t=ServerBase.ExecuteReq)

    if req.workflow_id not in self._workflows:
      raise ValueError(f'Workflow not found: {req.workflow_id}')

    upload_req: ServerBase.UploadWorkflowReq = self._workflows[req.workflow_id]
    error_context['upload_req'] = await error_context.LargeToFile(
        'upload_req', upload_req.model_dump(mode='json', round_trip=True))

    prov_spec: ProvisioningSpec = upload_req.prov_spec
    error_context['prov_spec'] = prov_spec.model_dump(mode='json',
                                                      round_trip=True)

    error_context['keepalive'] = req.keepalive
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
        ProvisionReq(id=req.job_id, bundle=prov_spec, keepalive=req.keepalive))
    error_context['provisioned'] = provisioned.model_dump(mode='json',
                                                          round_trip=True)
    logger.info('Successfully provisioned a ComfyUI instance',
                extra=error_context.Dump())
    ############################################################################
    keepalive_task = asyncio.create_task(
        self._KeepAlive(req.job_id, req.keepalive))
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
          return ServerBase.ExecuteSuccess(job_id=req.job_id,
                                           history=history,
                                           mapped_outputs=mapped_outputs)
    finally:
      keepalive_task.cancel()

  async def Execute(self, req: ExecuteReq) -> ExecuteRes:
    try:
      error_context = await _ErrorContext.Create(debug_path=self._debug_path,
                                                 key=req.job_id)
      error_context['req'] = await error_context.LargeToFile(
          'req', req.model_dump(mode='json', round_trip=True))
      # TODO: Make a blocking option that sends keepalives.
      # TODO: Make a nonblocking option that can be queried for status.

      success = await self._Execute(req, error_context=error_context.Copy())
      return Server.ExecuteRes(success=success, error=None)
    except Exception as e:
      error_id = uuid.uuid4().hex
      status_code = 500
      error_name = 'InternalError'
      message = f'An internal error occurred. Please report this error with the error_id: {json.dumps(error_id)}'
      if isinstance(e, _Error):
        status_code = 400
        error_name = e.error_name
        message = e.user_message
      error_context['error_id'] = error_id
      error_context['error_name'] = error_name
      error_context['status_code'] = status_code
      logger.exception('Execute failed',
                       exc_info=True,
                       stack_info=True,
                       extra={'error_context': error_context.Dump()})
      return Server.ExecuteRes(success=None,
                               error=Server.ExecuteError(
                                   error_id=error_id,
                                   status_code=status_code,
                                   name=error_name,
                                   message=message,
                                   context=error_context.UserDump(),
                               ))

  async def Touch(self, req: TouchReq) -> TouchRes:
    res: ProvisionerBase.TouchRes
    res = await self._provisioner.Touch(
        ProvisionerBase.TouchReq(id=req.job_id, keepalive=req.keepalive))
    return Server.TouchRes(success=res.success, message=res.message)
