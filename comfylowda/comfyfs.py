# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import asyncio
import io
from functools import partial
from typing import Any, Callable, Literal

import aiohttp
import fsspec
from comfy_catapult.api_client import ComfyAPIClient
from comfy_catapult.comfy_schema import ComfyUIPathTriplet
from comfy_catapult.remote_file_api_comfy import ComfySchemeURLToTriplet


class _Writable(fsspec.spec.AbstractBufferedFile):

  def __init__(self, done: Callable[[io.BytesIO], None]):
    self._buffer = io.BytesIO()
    self._done = done

  def write(self, data: bytes):
    self._buffer.write(data)

  def writelines(self, lines):
    return self._buffer.writelines(lines)

  def readable(self) -> bool:
    return False

  def seekable(self) -> bool:
    return False

  def writable(self) -> bool:
    return True

  def close(self) -> None:
    self._buffer.close()

  @property
  def closed(self) -> bool:
    return self._buffer.closed

  def __enter__(self) -> '_Writable':
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self._done(self._buffer)
    self.close()

  def seek(self, pos, whence=0):
    raise io.UnsupportedOperation('Seek operation not supported.')

  def tell(self):
    raise io.UnsupportedOperation('Tell operation not supported.')

  def read(self, *args, **kwargs):
    raise io.UnsupportedOperation('Read operation not supported.')

  def read1(self, *args, **kwargs):
    raise io.UnsupportedOperation('Read operation not supported.')

  def readinto(self, *args, **kwargs):
    raise io.UnsupportedOperation('Read operation not supported.')

  def readline(self, *args, **kwargs):
    raise io.UnsupportedOperation('Read operation not supported.')

  def readlines(self, *args, **kwargs):
    raise io.UnsupportedOperation('Read operation not supported.')


class _Readable(fsspec.spec.AbstractBufferedFile):

  def __init__(self, buffer: io.BytesIO):
    # Initialize the internal buffer with initial bytes
    self._buffer = buffer

  def read(self, size=-1):
    return self._buffer.read(size)

  def readline(self, size=-1):
    return self._buffer.readline(size)

  def readlines(self, hint=-1):
    return self._buffer.readlines(hint)

  def seek(self, offset, whence=0):
    raise io.UnsupportedOperation('Seek operation not supported.')

  def tell(self):
    raise io.UnsupportedOperation('Tell operation not supported.')

  def readable(self):
    return True

  def seekable(self) -> bool:
    return False

  def writable(self):
    return False

  def write(self, b):
    raise io.UnsupportedOperation('Write operation not supported.')

  def writelines(self, lines):
    raise io.UnsupportedOperation('Write operation not supported.')

  def flush(self):
    raise io.UnsupportedOperation(
        'Flush operation not supported on read-only buffer.')

  def close(self):
    return self._buffer.close()

  @property
  def closed(self):
    return self._buffer.closed

  def __enter__(self) -> '_Readable':
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()


class ComfyUISchemeFSAbstract(fsspec.spec.AbstractFileSystem):

  def __init__(self, protocol: Literal['http', 'https'], **kwargs):
    super().__init__(**kwargs)
    self._protocol = protocol

  async def _DoneWriting(self, buffer: io.BytesIO, *, printable_path: str,
                         comfy_api_url: str, triplet: ComfyUIPathTriplet):
    try:
      async with ComfyAPIClient(comfy_api_url) as api_client:
        await api_client.PostUploadImage(folder_type=triplet.type,
                                         subfolder=triplet.subfolder,
                                         filename=triplet.filename,
                                         data=buffer.getvalue(),
                                         overwrite=True)
    except Exception as e:
      raise IOError(f'Error writing file: {printable_path}') from e

  def _DoneWritingSync(self, buffer: io.BytesIO, *, printable_path: str,
                       comfy_api_url: str, triplet: ComfyUIPathTriplet):
    asyncio.run(
        self._DoneWriting(printable_path=printable_path,
                          comfy_api_url=comfy_api_url,
                          triplet=triplet,
                          buffer=buffer))

  def _ParsePath(self, path: Any) -> tuple[str, str, ComfyUIPathTriplet]:
    comfy_scheme_url = f'comfy+{self._protocol}://{path}'
    comfy_api_url, triplet = ComfySchemeURLToTriplet(comfy_scheme_url)
    return comfy_scheme_url, comfy_api_url, triplet

  def _ParseURL(self,
                comfy_scheme_url: str) -> tuple[str, str, ComfyUIPathTriplet]:
    comfy_api_url, triplet = ComfySchemeURLToTriplet(comfy_scheme_url)
    return comfy_scheme_url, comfy_api_url, triplet

  async def _open_async(self,
                        path,
                        mode='rb',
                        **kwargs) -> _Writable | _Readable:
    comfy_scheme_url, comfy_api_url, triplet = self._ParsePath(path)
    printable_path = comfy_scheme_url
    if mode == 'rb':
      try:
        async with ComfyAPIClient(comfy_api_url) as api_client:
          file_contents = await api_client.GetView(folder_type=triplet.type,
                                                   subfolder=triplet.subfolder,
                                                   filename=triplet.filename)
          return _Readable(io.BytesIO(file_contents))
      except aiohttp.ClientResponseError as e:
        if e.status == 404:
          raise FileNotFoundError(f'File not found: {printable_path}') from e
        raise
      except Exception as e:
        raise IOError(f'Error opening file: {printable_path}') from e

    if mode == 'wb':
      return _Writable(done=partial(self._DoneWritingSync,
                                    printable_path=printable_path,
                                    comfy_api_url=comfy_api_url,
                                    triplet=triplet))
    raise ValueError(f'Invalid mode: {mode}')

  def _open(self, path, mode='rb', **kwargs):
    return asyncio.run(self._open_async(path, mode, **kwargs))

  def ls(self, path, detail=False, **kwargs):
    raise NotImplementedError()

  async def _Exists(self, path: str) -> bool:
    comfy_scheme_url, comfy_api_url, triplet = self._ParseURL(path)
    try:
      async with ComfyAPIClient(comfy_api_url) as api_client:
        _ = await api_client.GetView(folder_type=triplet.type,
                                     subfolder=triplet.subfolder,
                                     filename=triplet.filename)
        return True
    except aiohttp.ClientResponseError as e:
      if e.status == 404:
        return False
      raise

  def exists(self, path, **kwargs) -> bool:
    if not isinstance(path, str):
      raise TypeError(f'path must be a string: {path}')
    return asyncio.run(self._Exists(path))


class ComfyUISchemeFSHTTP(ComfyUISchemeFSAbstract):

  def __init__(self, **kwargs):
    super().__init__(protocol='http', **kwargs)


class ComfyUISchemeFSHTTPS(ComfyUISchemeFSAbstract):

  def __init__(self, **kwargs):
    super().__init__(protocol='https', **kwargs)


def RegisterComfyUIFS():
  fsspec.register_implementation('comfy+http', ComfyUISchemeFSHTTP)
  fsspec.register_implementation('comfy+https', ComfyUISchemeFSHTTPS)


if __name__ == '__main__':
  fs = fsspec.filesystem('comfy+http')

  with fs.open('comfy+http://host.docker.internal:8188/temp/sub/sub2/1.txt',
               'wb') as f:
    f.write(b'Hello, World!')
  with fs.open('comfy+http://host.docker.internal:8188/temp/sub/sub2/1.txt',
               'rb') as f:
    print(f.read())
  print(fs.exists('comfy+http://host.docker.internal:8188/temp/sub/sub2/1.txt'))
