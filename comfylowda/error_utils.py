# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import sys
from datetime import datetime
from typing import Any, Dict

import yaml
from anyio import Path
from pydantic import BaseModel
from rich.console import Console
from slugify import slugify
from typing_extensions import TypeVar

from .base_types import Response, ResponseErrorBase
from .lowda_types import JSONSerializable


class _CustomDumper(yaml.SafeDumper):

  def represent_tuple(self, data):
    return self.represent_list(data)

  def represent_object(self, data):
    return self.represent_str(str(data))

  def literal_presenter(self, data):
    if '\n' in data:
      return self.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return self.represent_scalar('tag:yaml.org,2002:str', data)

  def represent_unserializable(self, data):
    return self.represent_str(f'unserializable: {str(data)}')


_CustomDumper.add_representer(tuple, _CustomDumper.represent_tuple)
_CustomDumper.add_representer(object, _CustomDumper.represent_object)
_CustomDumper.add_representer(str, _CustomDumper.literal_presenter)
_CustomDumper.add_multi_representer(object,
                                    _CustomDumper.represent_unserializable)


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
    # TODO: dereferences large files.
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


# TODO: Make a neat converter back and forth between _Error and
# ResponseErrorBase.
# TODO: Make a utility to propagate errors, e.g if a server call fails because
# of a call to another server.
class _Error(Exception):

  def __init__(self, *, user_message: str, internal_message: str,
               status_code: int | None, error_name: str,
               internal_context: dict | None, io_name: str | None) -> None:
    super().__init__(internal_message)
    self.user_message = user_message
    self.internal_message = internal_message
    self.status_code = status_code
    self.error_name = error_name
    self.internal_context = internal_context
    self.io_name = io_name


class _HTTPError(Exception):

  def __init__(self, status: int, reason: str | None,
               response: str | None) -> None:
    self.status: int = status
    self.reason: str | None = reason
    self.response: str | None = response


################################################################################
_SuccessT = TypeVar('_SuccessT', bound=BaseModel)
_ErrorT = TypeVar('_ErrorT', bound=ResponseErrorBase)


async def _SuccessOrError(action: str, res: Response[_SuccessT,
                                                     _ErrorT]) -> _SuccessT:
  if res.error is not None:
    error: _ErrorT = res.error
    raise _Error(user_message=f'Failed to {action}. {error.message}',
                 internal_message=f'Failed to {action}.',
                 status_code=error.status_code,
                 error_name=error.name,
                 internal_context=error.context,
                 io_name=error.error_id)
  if res.success is None:
    raise _Error(
        user_message='Internal Error.',
        internal_message=
        'res.success is None, and res.error is None. This should not happen.',
        status_code=500,
        error_name='InternalError',
        internal_context={
            'res': res.model_dump(mode='json', round_trip=True, by_alias=True)
        },
        io_name='InternalError')
  return res.success


async def _SuccessOrExit(action: str, res: Response[_SuccessT, _ErrorT],
                         console: Console) -> _SuccessT:
  if res.error is not None:
    error: _ErrorT = res.error
    console.print(f'Failed to {action}. Exiting.', style='red')
    console.print('Error:',
                  _YamlDump(
                      error.model_dump(mode='json',
                                       round_trip=True,
                                       by_alias=True)),
                  style='red')
    console.print(f'Failed to {action}. Exiting.', style='red')
    sys.exit(1)
  if res.success is None:
    raise AssertionError(
        'res.success is None, and res.error is None. This should not happen.')
  return res.success
