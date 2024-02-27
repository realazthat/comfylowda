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
from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import Annotated, Any, Dict, Hashable, List, Literal, NamedTuple
from urllib.parse import urlparse

import aiofiles
import fsspec
import pydantic
import pydash
from anyio import Path
from comfy_catapult.api_client import ComfyAPIClient
from comfy_catapult.catapult import ComfyCatapult
from comfy_catapult.comfy_schema import (APIHistoryEntry, APINodeID,
                                         APIObjectInfo, APIOutputUI,
                                         APIWorkflow, ComfyFolderType,
                                         ComfyUIPathTriplet)
from comfy_catapult.comfy_utils import WatchVar
from pydantic import BaseModel, Field
from slugify import slugify

from comfylowda.comfyfs import _Writable

from .comfy_schema import Workflow

logger = logging.getLogger(__name__)

PublicURL = Annotated[str, 'PublicURL']
LocalURL = Annotated[str, 'LocalURL']

# JSONSerializable = Union[str, int, float, bool, None, Dict[str,
#                                                            'JSONSerializable'],
#                          List['JSONSerializable'], Tuple['JSONSerializable',
#                                                          ...]]
JSONSerializable = Any
JSON_SERIALIZABLE_TYPES = (str, int, float, bool, type(None), dict, list, tuple)


def IsJSONSerializable(value: Any) -> bool:
  if isinstance(value, JSON_SERIALIZABLE_TYPES):
    return True
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
  return False


class IOSpec(NamedTuple):
  io_url: str
  """This could be a file URI, or any supported protocol.
  
  Additionally, it can be a comfy+http or comfy+https URL in the form of:
  comfy+https://comfy-server-host:port/folder_type/subfolder/sub/sub/filename
  
  This URL can be used to upload and download files to and from the comfy
  server.

  If the comfy+http or comfy+https URL is used, then the ComfyUI API will
  directly be used for upload and download for all upload/download operations.
  """
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

  class Overwrite(enum.Enum):
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

  def AddFS(self, uri_prefix: str, fs: fsspec.spec.AbstractFileSystem,
            mode: Mode):

    self._fs[mode][uri_prefix] = fs
    if mode == 'rw':
      self._fs['r'][uri_prefix] = fs
      self._fs['w'][uri_prefix] = fs

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
    for uri_prefix, fs in self._fs[mode].items():
      if uri.startswith(uri_prefix):
        return fs
    raise ValueError(f'No matching FS for {uri}, self._fs: {list(self._fs)}')


async def DownloadURLToB64(*, remote: RemoteFileAPIBase, io_spec: IOSpec,
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


class ComfyRemoteInfo(BaseModel):
  comfy_api_url: str
  # {prefix => URL}
  upload: Dict[str, IOSpec]
  download: Dict[str, IOSpec]


def GetRemoteIOSpec(*, path: str, prefix2iospec: Dict[str, IOSpec]) -> IOSpec:
  for prefix, io_spec in prefix2iospec.items():
    if path.startswith(prefix):

      return io_spec._replace(io_url=io_spec.io_url + path[len(prefix):])
  raise ValueError(
      f'No matching prefix for {path}, prefix2url: {prefix2iospec}')


class ProvisioningBundle(BaseModel):
  files: Dict[str, PublicURL | LocalURL]
  archives: Dict[str, PublicURL | LocalURL]
  custom_nodes: Dict[str, PublicURL | LocalURL]


class LowdaInputFieldType(enum.Enum):
  LITERAL = enum.auto()
  """Anything JSON Serializable. Just gets passed through."""
  JSON_STR = enum.auto()
  """A JSON string. Gets deserialized."""
  FILE_B64 = enum.auto()
  """JSON Serializable dict that will be validated using JSONFileB64."""
  TRIPLET_FILE_B64 = enum.auto()
  """JSON Serializable dict that will be validated using JSONTripletB64."""


class JSONFileB64(BaseModel):
  filename: str
  content_b64: str


class JSONFileTripletB64(BaseModel):
  folder_type: ComfyFolderType
  subfolder: str
  filename: str
  content_b64: str


class LowdaOutputFieldType(enum.Enum):
  LITERAL = enum.auto()
  JSON_STR = enum.auto()
  FILE_B64 = enum.auto()
  TRIPLET_FILE_B64 = enum.auto()


class LowdaInputMapping(BaseModel):
  node_id: APINodeID = Field(
      ...,
      description='The node_id of a node in the ComfyUI Workflow API format.'
      ' See https://github.com/realazthat/comfylowda/assets/sdxlturbo_example_api.json for an example Workflow API format'
  )
  comfy_api_field_path: str = Field(
      ...,
      description=
      'A pydash field path, for the pydash.get() and pydash.set_() functions.'
      ' The field_path begins at the .inputs field of a node in the ComfyUI Workflow API format.'
      ' See https://github.com/realazthat/comfylowda/assets/sdxlturbo_example_api.json for an example Workflow API format'
  )
  comfy_api_field_type: LowdaInputFieldType = Field(
      ..., description='The type of the field.')


class LowdaOutputMapping(BaseModel):
  """Defines a mapping from the /history/{prompt_id} entry to a field in the user output.
  """
  node_id: APINodeID
  comfy_api_field_path: str
  comfy_api_field_type: LowdaOutputFieldType


class WorkflowTemplateBundle(BaseModel):
  workflow_template: Workflow
  api_workflow_template: APIWorkflow
  important: List[APINodeID]
  object_info: APIObjectInfo
  user_input_mappings: Dict[str, LowdaInputMapping]
  user_output_mappings: Dict[str, LowdaOutputMapping]


class WorkflowBundle(BaseModel):
  template_bundle: WorkflowTemplateBundle
  user_input_values: Dict[str, JSONSerializable]


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


class ManagerBase(ABC):

  # @abstractmethod
  # async def Serve(self, workflow: WorkflowTemplateBundle, endpoint_path: str) -> None:
  #   raise NotImplementedError()

  class ExecuteReq(BaseModel):
    job_id: str

    @pydantic.field_validator('job_id')
    @classmethod
    def check_slugified(cls, v):
      if slugify(v) != v:
        raise ValueError('job_id must be slugified; use python-slugify')
      return v

    workflow: WorkflowBundle
    provisioning: ProvisioningBundle
    keepalive: float

  class ExecuteRes(BaseModel):
    job_id: str
    history: APIHistoryEntry
    mapped_outputs: Dict[str, JSONSerializable]

  class TouchReq(BaseModel):
    job_id: str
    keepalive: float

  class TouchRes(BaseModel):
    success: bool
    message: str

  @abstractmethod
  async def Execute(self, req: ExecuteReq) -> ExecuteRes:
    raise NotImplementedError()

  @abstractmethod
  async def Touch(self, req: TouchReq) -> TouchRes:
    raise NotImplementedError()


def TripletDictToTriplet(triplet_dict: dict) -> ComfyUIPathTriplet:

  # file_dict: dict = pydash.get(node_outputs.root, field_path)

  if 'filename' not in triplet_dict:
    raise Exception(f'Expected "filename" in {triplet_dict}')
  filename: str = triplet_dict['filename']
  if not isinstance(filename, str):
    raise Exception(f'Expected "filename" to be str, got {type(filename)}')
  if 'subfolder' not in triplet_dict:
    raise Exception(f'Expected "subfolder" in {triplet_dict}')
  subfolder: str = triplet_dict['subfolder']
  if not isinstance(subfolder, str):
    raise Exception(f'Expected "subfolder" to be str, got {type(subfolder)}')
  if 'type' not in triplet_dict:
    raise Exception(f'Expected "type" in {triplet_dict}')
  folder_type: Literal['temp', 'output'] = triplet_dict['type']
  if not isinstance(folder_type, str):
    raise Exception(f'Expected "type" to be str, got {type(folder_type)}')
  if folder_type not in ['temp', 'output']:
    raise Exception(
        f'Expected "type" to be "temp" or "output", got {folder_type}')

  return ComfyUIPathTriplet(type=folder_type,
                            subfolder=subfolder,
                            filename=filename)


def NodeOutputToTriplets(node_id: APINodeID, job_history: APIHistoryEntry,
                         field_path: Hashable | List[Hashable],
                         comfy_api_url: str) -> List[ComfyUIPathTriplet]:
  """
  * field_path=='gifs' works for Video Combine
  * field_path=='images' works for Preview Image



  Example Preview Image Workflow API json:

  ```
  '25':
    inputs:
      images:
      - '8'
      - 0
    class_type: PreviewImage
    meta:
      title: Preview Image
  ```

  Example Preview Image node output:

  ```
  outputs:
    '25':
      images:
      - filename: ComfyUI_temp_huntb_00001_.png
        subfolder: ''
        type: temp
  ```

  Example Video Combine node output:

  TODO: Put an example here.

  Args:
      node_id: The node_id.
      job_history: The job_history.
      field_path: A pydash field path, for the pydash.get() and pydash.set_()
        functions.
      comfy_api_url: e.g http://127.0.0.1:8188.
  """
  if job_history.outputs is None:
    raise AssertionError('job_history.outputs is None')

  if node_id not in job_history.outputs:
    raise Exception(f'{node_id} not in job_history.outputs')

  node_outputs: APIOutputUI = job_history.outputs[node_id]

  triplet_dicts: List[dict] = pydash.get(node_outputs.root, field_path)
  if not isinstance(triplet_dicts, (list, tuple)):
    raise Exception(
        f'Expected triplet_dicts to be list, got {type(triplet_dicts)}')

  return [TripletDictToTriplet(triplet_dict) for triplet_dict in triplet_dicts]


def NodeOutputToTriplet(node_id: APINodeID, job_history: APIHistoryEntry,
                        field_path: Hashable | List[Hashable],
                        comfy_api_url: str) -> ComfyUIPathTriplet:
  """
  * field_path=='gifs[0]' works for Video Combine
  * field_path=='images[0]' works for Preview Image



  Example Preview Image Workflow API json:

  ```
  '25':
    inputs:
      images:
      - '8'
      - 0
    class_type: PreviewImage
    meta:
      title: Preview Image
  ```

  Example Preview Image node output:

  ```
  outputs:
    '25':
      images:
      - filename: ComfyUI_temp_huntb_00001_.png
        subfolder: ''
        type: temp
  ```

  Example Video Combine node output:

  TODO: Put an example here.

  Args:
      node_id: The node_id.
      job_history: The job_history.
      field_path: A pydash field path, for the pydash.get() and pydash.set_()
        functions.
      comfy_api_url: e.g http://127.0.0.1:8188.
  """
  if job_history.outputs is None:
    raise AssertionError('job_history.outputs is None')

  if node_id not in job_history.outputs:
    raise Exception(f'{node_id} not in job_history.outputs')

  node_outputs: APIOutputUI = job_history.outputs[node_id]

  triplet_dict: dict = pydash.get(node_outputs.root, field_path)
  if not isinstance(triplet_dict, (dict)):
    raise Exception(
        f'Expected triplet_dict to be dict, got {type(triplet_dict)}')

  return TripletDictToTriplet(triplet_dict)


def GetTripletFileIOSpec(*, triplet: ComfyUIPathTriplet,
                         remote_info: ComfyRemoteInfo,
                         mode: Literal['upload', 'download']) -> IOSpec:
  remote_path = triplet.ToLocalPathStr(include_folder_type=True)
  if mode == 'download':
    download_io_spec = GetRemoteIOSpec(path=remote_path,
                                       prefix2iospec=remote_info.download)
    return download_io_spec
  elif mode == 'upload':
    upload_io_spec = GetRemoteIOSpec(path=remote_path,
                                     prefix2iospec=remote_info.upload)
    return upload_io_spec
  else:
    raise ValueError(f'Unsupported mode: {json.dumps(mode)}')


async def DownloadTripletFile(*, triplet: ComfyUIPathTriplet,
                              remote_info: ComfyRemoteInfo,
                              remote: RemoteFileAPIBase,
                              trusted_dst_path: Path):
  download_io_spec = GetTripletFileIOSpec(triplet=triplet,
                                          remote_info=remote_info,
                                          mode='download')
  await remote.DownloadFile(untrusted_src_io_spec=download_io_spec,
                            trusted_dst_path=trusted_dst_path)


class UserAPIInputFieldDecoderBase(ABC):

  @abstractmethod
  async def Decode(
      self, comfy_info: ComfyRemoteInfo,
      comfy_api_field_type: LowdaInputFieldType,
      user_input_encoded_value: JSONSerializable) -> JSONSerializable:
    raise NotImplementedError()


class LiteralInputFieldDecoder(UserAPIInputFieldDecoderBase):

  async def Decode(
      self, comfy_info: ComfyRemoteInfo,
      comfy_api_field_type: LowdaInputFieldType,
      user_input_encoded_value: JSONSerializable) -> JSONSerializable:
    if not isinstance(user_input_encoded_value, JSON_SERIALIZABLE_TYPES):
      raise ValueError(
          'Expected user_input_encoded_value to be one of '
          f'{JSON_SERIALIZABLE_TYPES}, because '
          f'LowdaInputMapping.comfy_api_field_type={json.dumps(comfy_api_field_type)}, '
          f'got {type(user_input_encoded_value)}')
    if not IsJSONSerializable(user_input_encoded_value):
      raise ValueError(
          'Expected user_input_encoded_value to be JSONSerializable, because '
          f'LowdaInputMapping.comfy_api_field_type={json.dumps(comfy_api_field_type)}, '
          f'got {type(user_input_encoded_value)}')
    return user_input_encoded_value


class JSONStrInputFieldDecoder(UserAPIInputFieldDecoderBase):

  async def Decode(
      self, comfy_info: ComfyRemoteInfo,
      comfy_api_field_type: LowdaInputFieldType,
      user_input_encoded_value: JSONSerializable) -> JSONSerializable:
    if not isinstance(user_input_encoded_value, (str)):
      raise ValueError(
          'Expected user_input_encoded_value to be str because '
          f'LowdaInputMapping.comfy_api_field_type={repr(comfy_api_field_type)}, '
          f'got {type(user_input_encoded_value)}')
    user_input_value = json.loads(user_input_encoded_value)
    if not isinstance(user_input_value, JSON_SERIALIZABLE_TYPES):
      raise AssertionError(
          'Expected user_input_encoded_value==json.loads(user_input_encoded_value) to be one of '
          f'{JSON_SERIALIZABLE_TYPES}, because '
          f'LowdaInputMapping.comfy_api_field_type={repr(comfy_api_field_type)}, '
          f'got {type(user_input_encoded_value)}')
    return user_input_value


class UserAPIOutputFieldEncoderBase(ABC):

  @abstractmethod
  async def Encode(self, comfy_info: ComfyRemoteInfo,
                   comfy_api_field_type: LowdaOutputFieldType,
                   user_output_value: JSONSerializable) -> JSONSerializable:
    raise NotImplementedError()

  def Validate(self, user_output_value: JSONSerializable):
    if not isinstance(user_output_value, JSON_SERIALIZABLE_TYPES):
      raise ValueError('Expected user_output_value to be one of '
                       f'{JSON_SERIALIZABLE_TYPES}, '
                       f'got {type(user_output_value)}')
    if not IsJSONSerializable(user_output_value):
      raise ValueError('Expected IsJSONSerializable(user_output_value)==True')


class LiteralOutputFieldEncoder(UserAPIOutputFieldEncoderBase):

  async def Encode(self, comfy_info: ComfyRemoteInfo,
                   comfy_api_field_type: LowdaOutputFieldType,
                   user_output_value: JSONSerializable) -> JSONSerializable:
    self.Validate(user_output_value)
    return user_output_value


class JSONStrOutputFieldEncoder(UserAPIOutputFieldEncoderBase):

  async def Encode(self, comfy_info: ComfyRemoteInfo,
                   comfy_api_field_type: LowdaOutputFieldType,
                   user_output_value: JSONSerializable) -> JSONSerializable:
    self.Validate(user_output_value)
    return json.dumps(user_output_value)


class TripletB64OutputFieldEncoder(UserAPIOutputFieldEncoderBase):

  def __init__(self, *, remote: RemoteFileAPIBase, tmp_dir_path: Path):
    self._remote = remote
    self._tmp_dir_path = tmp_dir_path

  async def Encode(self, comfy_info: ComfyRemoteInfo,
                   comfy_api_field_type: LowdaOutputFieldType,
                   user_output_value: JSONSerializable) -> JSONSerializable:
    self.Validate(user_output_value)
    triplet = ComfyUIPathTriplet.model_validate(user_output_value)
    download_io_spec = GetTripletFileIOSpec(triplet=triplet,
                                            remote_info=comfy_info,
                                            mode='download')
    return await DownloadURLToB64(remote=self._remote,
                                  io_spec=download_io_spec,
                                  tmp_dir_path=self._tmp_dir_path)


class Manager(ManagerBase):
  ExecuteReq = ManagerBase.ExecuteReq
  ExecuteRes = ManagerBase.ExecuteRes

  def __init__(self, *, provisioner: ProvisionerBase, remote: RemoteFileAPIBase,
               tmp_dir_path: Path, debug_path: Path,
               debug_save_all: bool) -> None:
    self._provisioner = provisioner
    self._remote = remote
    self._tmp_dir_path = tmp_dir_path
    self._input_value_decoders: Dict[LowdaInputFieldType,
                                     UserAPIInputFieldDecoderBase] = {}
    self._output_value_encoders: Dict[LowdaOutputFieldType,
                                      UserAPIOutputFieldEncoderBase] = {}
    self._debug_path: Path = debug_path
    self._debug_save_all = debug_save_all

    self._input_value_decoders[
        LowdaInputFieldType.LITERAL] = LiteralInputFieldDecoder()
    self._input_value_decoders[
        LowdaInputFieldType.JSON_STR] = JSONStrInputFieldDecoder()
    self._output_value_encoders[
        LowdaOutputFieldType.LITERAL] = LiteralOutputFieldEncoder()
    self._output_value_encoders[
        LowdaOutputFieldType.JSON_STR] = JSONStrOutputFieldEncoder()
    self._output_value_encoders[
        LowdaOutputFieldType.TRIPLET_FILE_B64] = TripletB64OutputFieldEncoder(
            remote=self._remote, tmp_dir_path=self._tmp_dir_path)

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
                         workflow: WorkflowBundle,
                         prepared_workflow: APIWorkflow,
                         user_input_mapping: LowdaInputMapping,
                         user_input_encoded_value: JSONSerializable) -> None:
    node_id = user_input_mapping.node_id
    comfy_api_field_type: LowdaInputFieldType = user_input_mapping.comfy_api_field_type
    if node_id not in prepared_workflow.root:
      raise ValueError(f'{node_id} not in prepared_workflow.root')
    if comfy_api_field_type not in self._input_value_decoders:
      raise ValueError(
          f'comfy_api_field_type={comfy_api_field_type} not in self._input_value_decoders'
      )
    decoder = self._input_value_decoders[comfy_api_field_type]
    user_input_value: JSONSerializable
    user_input_value = await decoder.Decode(
        comfy_info=comfy_info,
        comfy_api_field_type=comfy_api_field_type,
        user_input_encoded_value=user_input_encoded_value)
    # TODO: Ensure that this field path exists.
    pydash.set_(prepared_workflow.root, user_input_mapping.comfy_api_field_path,
                user_input_value)

  async def _UploadValues(self, comfy_info: ComfyRemoteInfo,
                          workflow: WorkflowBundle,
                          prepared_workflow: APIWorkflow) -> None:
    user_input_mappings: Dict[str, LowdaInputMapping]
    user_input_mappings = workflow.template_bundle.user_input_mappings
    user_input_values = workflow.user_input_values
    for user_input_name, user_input_mapping in user_input_mappings.items():
      try:
        user_input_encoded_value = user_input_values[user_input_name]

        await self._UploadValue(
            comfy_info=comfy_info,
            workflow=workflow,
            prepared_workflow=prepared_workflow,
            user_input_mapping=user_input_mapping,
            user_input_encoded_value=user_input_encoded_value)
      except Exception as e:
        logger.exception('UploadValue failed',
                         exc_info=True,
                         stack_info=True,
                         extra={
                             'user_input_name': user_input_name,
                             'user_input_mapping': user_input_mapping,
                             'user_input_encoded_value':
                             user_input_encoded_value
                         })
        raise ValueError(
            f'Failed to upload {json.dumps(user_input_name)}={json.dumps(user_input_encoded_value)}'
        ) from e

  async def _DownloadValue(
      self, comfy_info: ComfyRemoteInfo, workflow: WorkflowBundle,
      prepared_workflow: APIWorkflow, history: APIHistoryEntry,
      user_output_mapping: LowdaOutputMapping) -> JSONSerializable:
    node_id = user_output_mapping.node_id
    comfy_api_field_type: LowdaOutputFieldType = user_output_mapping.comfy_api_field_type
    if history.outputs is None or node_id not in history.outputs:
      raise ValueError(f'{node_id} not in history.outputs')

    node_outputs: APIOutputUI = history.outputs[node_id]
    user_output_value: JSONSerializable
    user_output_value_any: Any = pydash.get(
        node_outputs.root, user_output_mapping.comfy_api_field_path)

    if not isinstance(user_output_value_any, JSON_SERIALIZABLE_TYPES):
      raise ValueError(
          f'Expected user_output_value to be one of {JSON_SERIALIZABLE_TYPES}, got {type(user_output_value_any)}'
      )
    if not IsJSONSerializable(user_output_value_any):
      raise ValueError(
          f'Expected user_output_value to be JSONSerializable, got {type(user_output_value_any)}'
      )
    user_output_value = user_output_value_any
    if comfy_api_field_type not in self._output_value_encoders:
      raise ValueError(
          f'comfy_api_field_type={comfy_api_field_type} not in self._output_value_encoders'
      )
    encoder = self._output_value_encoders[comfy_api_field_type]
    return await encoder.Encode(comfy_info=comfy_info,
                                comfy_api_field_type=comfy_api_field_type,
                                user_output_value=user_output_value)

  async def _DownloadValues(
      self, comfy_info: ComfyRemoteInfo, workflow: WorkflowBundle,
      prepared_workflow: APIWorkflow,
      history: APIHistoryEntry) -> Dict[str, JSONSerializable]:
    user_output_values: Dict[str, JSONSerializable] = {}

    user_output_name: str
    user_output_mapping: LowdaOutputMapping
    for user_output_name, user_output_mapping in workflow.template_bundle.user_output_mappings.items(
    ):
      try:
        user_output_values[user_output_name] = await self._DownloadValue(
            comfy_info=comfy_info,
            workflow=workflow,
            prepared_workflow=prepared_workflow,
            history=history,
            user_output_mapping=user_output_mapping,
        )
      except Exception as e:
        logger.exception('DownloadValue failed',
                         exc_info=True,
                         stack_info=True,
                         extra={
                             'user_output_name': user_output_name,
                             'user_output_mapping': user_output_mapping
                         })
        raise ValueError(
            f'Failed to download {json.dumps(user_output_name)}') from e

    return user_output_values

  async def _Execute(self, req: ExecuteReq) -> ExecuteRes:
    ProvisionReq = ProvisionerBase.ProvisionReq

    provisioned: ProvisionerBase.ProvisionRes
    provisioned = await self._provisioner.Provision(
        ProvisionReq(id=req.job_id,
                     bundle=req.provisioning,
                     keepalive=req.keepalive))
    keepalive_task = asyncio.create_task(
        self._KeepAlive(req.job_id, req.keepalive))

    try:
      async with ComfyAPIClient(
          comfy_api_url=provisioned.comfy_info.comfy_api_url) as client:
        async with ComfyCatapult(
            comfy_client=client,
            debug_path=self._debug_path,
            debug_save_all=self._debug_save_all) as catapult:
          api_workflow_template = req.workflow.template_bundle.api_workflow_template
          prepared_workflow = api_workflow_template.model_copy(deep=True)

          await self._UploadValues(comfy_info=provisioned.comfy_info,
                                   workflow=req.workflow,
                                   prepared_workflow=prepared_workflow)
          history_dict = await catapult.Catapult(
              job_id=req.job_id,
              prepared_workflow=prepared_workflow.model_dump(),
              important=req.workflow.template_bundle.important)
          history: APIHistoryEntry = APIHistoryEntry.model_validate(
              history_dict)

          mapped_outputs: Dict[str, JSONSerializable]
          mapped_outputs = await self._DownloadValues(
              comfy_info=provisioned.comfy_info,
              workflow=req.workflow,
              prepared_workflow=prepared_workflow,
              history=history)

          return self.ExecuteRes(job_id=req.job_id,
                                 history=history,
                                 mapped_outputs=mapped_outputs)
    finally:
      keepalive_task.cancel()

  async def Execute(self, req: ExecuteReq) -> ExecuteRes:
    return await self._Execute(req)

  async def Touch(self, req: ManagerBase.TouchReq) -> ManagerBase.TouchRes:
    res: ProvisionerBase.TouchRes
    res = await self._provisioner.Touch(
        ProvisionerBase.TouchReq(id=req.job_id, keepalive=req.keepalive))
    return ManagerBase.TouchRes(success=res.success, message=res.message)
