# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

from datetime import datetime
from typing import Any, Dict

import yaml
from anyio import Path
from slugify import slugify

from .lowda_types import JSONSerializable


class _CustomDumper(yaml.Dumper):

  def represent_tuple(self, data):
    return self.represent_list(data)

  def represent_object(self, data):
    return self.represent_str(str(data))

  def literal_presenter(self, data):
    if '\n' in data:
      return self.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return self.represent_scalar('tag:yaml.org,2002:str', data)


_CustomDumper.add_representer(tuple, _CustomDumper.represent_tuple)
_CustomDumper.add_representer(object, _CustomDumper.represent_object)
_CustomDumper.add_representer(str, _CustomDumper.literal_presenter)


def _YamlDump(data: Any) -> str:
  return yaml.dump(data,
                   indent=2,
                   Dumper=_CustomDumper,
                   sort_keys=False,
                   allow_unicode=True,
                   default_flow_style=False,
                   width=80)


class _LargeFileReference:
  """Used for internal purposes only, for tracking large files by _ErrorContext."""

  def __init__(self, file_uri: str):
    self.file_uri = file_uri


class _ErrorContext:

  @classmethod
  async def Create(cls, *, debug_path: Path, key: str) -> '_ErrorContext':
    return cls(debug_path=await debug_path.absolute(),
               key=key,
               _private_use_create_instead=cls._PrivateUseCreateInstead())

  class _PrivateUseCreateInstead:
    pass

  def __init__(self, *, debug_path: Path, key: str,
               _private_use_create_instead: '_PrivateUseCreateInstead'):
    self._debug_path = debug_path
    self._key = key
    self._error_context_path = self._debug_path / slugify(self._key)
    self._context: Dict[str, JSONSerializable | _LargeFileReference] = {}

  async def LargeToFile(self, name: str,
                        msg: JSONSerializable) -> _LargeFileReference:
    directory = self._error_context_path / slugify(datetime.now().isoformat())
    await directory.mkdir(parents=True, exist_ok=True)
    path = directory / slugify(name)
    yaml_path = path.with_suffix('.yaml')
    await yaml_path.write_text(_YamlDump(msg))
    if not yaml_path.is_absolute():
      yaml_path = await yaml_path.absolute()
    return _LargeFileReference(file_uri=yaml_path.as_uri())

  def Add(self, *, name: str, msg: JSONSerializable) -> None:
    self._context[name] = msg

  def Dump(self) -> JSONSerializable:
    # TODO: Mark the large files to have a longer TTL if it is being dumped.
    dumped = self._context.copy()
    for k, v in dumped.items():
      if isinstance(v, _LargeFileReference):
        dumped[k] = v.file_uri
    return dumped

  def UserDump(self) -> JSONSerializable:
    """Same as Dump() but avoids information that is marked as internal-only, and dereferences large files."""
    # TODO: Add the ability to mark internal-only keys.
    dumped = self._context.copy()
    for k, v in dumped.items():
      if isinstance(v, _LargeFileReference):
        dumped[k] = v.file_uri
    return dumped

  def Copy(self) -> '_ErrorContext':
    copy_cxt = _ErrorContext(
        debug_path=self._debug_path,
        key=self._key,
        _private_use_create_instead=self._PrivateUseCreateInstead())
    copy_cxt._context = self._context.copy()
    return copy_cxt

  def __setitem__(self, key: str, value: JSONSerializable) -> None:
    self._context[key] = value
