# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import asyncio
import functools
import logging
import textwrap
import uuid
from json import JSONDecodeError
from typing import List, Tuple

import aiohttp
import httpx
import requests
from anyio import Path
from pydantic import BaseModel
from rich.console import Console
from typing_extensions import TypeVar

from .error_utils import _ErrorContext, _HTTPError, _YamlDump
from .lowda import ServerBase
from .web.routes import (JOB_EXECUTE_ENDPOINT, JOB_TOUCH_ENDPOINT,
                         WORKFLOW_DOWNLOAD_ENDPOINT, WORKFLOW_UPLOAD_ENDPOINT)

logger = logging.getLogger(__name__)
_RequestModelT = TypeVar('_RequestModelT', bound=BaseModel)
_ResponseModelT = TypeVar('_ResponseModelT', bound=BaseModel)


async def _ClientPostWithRequests(action: str, url: str, req: BaseModel,
                                  resp_type: type[_ResponseModelT],
                                  error_context: _ErrorContext,
                                  console: Console) -> _ResponseModelT:
  response: requests.Response
  response = await asyncio.to_thread(requests.post,
                                     url,
                                     json=req.model_dump(mode='json',
                                                         round_trip=True,
                                                         by_alias=True),
                                     timeout=100)

  error_context['response.headers'] = dict(response.headers)
  error_context['response.status_code'] = response.status_code
  error_context['response.reason'] = response.reason
  error_context['response.ok'] = response.ok
  if not response.ok:
    response_text: str | None = None
    try:
      response_json_dict = await asyncio.to_thread(response.json)
      error_context['response_json_dict'] = response_json_dict
      response_text = _YamlDump(response_json_dict)
      error_context['response_json_text_yaml'] = response_text
    except JSONDecodeError as e:
      error_context['response_json_dict_error'] = str(e)
    if response_text is None:
      response_text = response.text
    error_context['response_text'] = response_text

    console.print(f'Failed to {action}.', style='red')
    console.print('Headers:\n' +
                  textwrap.indent(_YamlDump(dict(response.headers)), '  '),
                  style='red')
    console.print('Error context:\n' +
                  textwrap.indent(_YamlDump(error_context.Dump()), '  '),
                  style='red')
    console.print(f'Failed to {action}. Exiting.', style='red')
    raise _HTTPError(status=response.status_code,
                     reason=response.reason,
                     response=response_text)
  response_dict = await asyncio.to_thread(response.json)
  return resp_type.model_validate(response_dict)


async def _ClientPostWithHTTPX(action: str, url: str, req: BaseModel,
                               resp_type: type[_ResponseModelT],
                               error_context: _ErrorContext,
                               console: Console) -> _ResponseModelT:
  async with httpx.AsyncClient() as client:
    response = await client.post(url,
                                 json=req.model_dump(mode='json',
                                                     round_trip=True,
                                                     by_alias=True))

    error_context['response.http_version'] = response.http_version
    error_context['response.headers'] = response.headers.multi_items()
    error_context['response.status_code'] = response.status_code
    error_context['response.reason_phrase'] = response.reason_phrase
    error_context['response.is_success'] = response.is_success
    error_context['response.is_error'] = response.is_closed
    error_context['response.is_error'] = response.is_error
    error_context['response.is_server_error'] = response.is_server_error
    error_context['response.is_client_error'] = response.is_client_error
    if not response.is_success:
      try:
        response_dict = await response.json()
        error_context['response_dict'] = response_dict
      except JSONDecodeError as e:
        error_context['response_dict_error'] = str(e)
      console.print(f'Failed to {action}.', style='red')
      console.print(
          'Headers:\n' +
          textwrap.indent(_YamlDump(response.headers.multi_items()), '  '),
          style='red')
      console.print('Error context:\n' +
                    textwrap.indent(_YamlDump(error_context.Dump()), '  '),
                    style='red')
      console.print(f'Failed to {action}. Exiting.', style='red')
      raise _HTTPError(status=response.status_code,
                       reason=response.reason_phrase,
                       response=response.text)
    response_dict = await response.json()
    return resp_type.model_validate(response_dict)


async def _ClientPostWithAIOHttp(action: str, url: str, req: BaseModel,
                                 resp_type: type[_ResponseModelT],
                                 error_context: _ErrorContext,
                                 console: Console) -> _ResponseModelT:
  async with aiohttp.ClientSession() as client:

    async with await client.post(url,
                                 json=req.model_dump(
                                     mode='json',
                                     round_trip=True,
                                     by_alias=True)) as response:
      headers_list: List[Tuple[str, str]] = []
      for k, v in response.headers.items():
        headers_list.append((str(k), v))
      error_context['response.version'] = (tuple(
          response.version) if response.version is not None else None)
      error_context['response.headers'] = headers_list
      error_context['response.status_code'] = response.status
      error_context['response.reason_phrase'] = response.reason
      error_context['response.ok'] = response.ok
      if not response.ok:
        response_text: str | None = None
        try:
          response_dict = await response.json()
          error_context['response_dict'] = response_dict
          response_text = _YamlDump(response_dict)
          error_context['response_dict_yaml'] = response_text
        except Exception as e:
          error_context['response_dict_error'] = str(e)

        if response_text is None:
          try:
            response_text = await response.text()
            error_context['response_text'] = response_text
          except Exception as e:
            error_context['response_text_error'] = str(e)
        console.print(f'Failed to {action}.', style='red')
        console.print('Headers:\n' +
                      textwrap.indent(_YamlDump(headers_list), '  '),
                      style='red')
        console.print('Error context:\n' +
                      textwrap.indent(_YamlDump(error_context.Dump()), '  '),
                      style='red')
        console.print(f'Failed to {action}. Exiting.', style='red')
        raise _HTTPError(status=response.status,
                         reason=response.reason,
                         response=response_text)
      response_dict = await response.json()
      return resp_type.model_validate(response_dict)


def _ConstructEndpoint(*, lowda_url: str, endpoint: str) -> str:
  if lowda_url.endswith('/'):
    lowda_url = lowda_url[:-1]
  return lowda_url + '/' + endpoint


def _WrapError(action: str, ResponseModelT, ErrorModelT):

  def _WrapErrorDecorator(func):

    @functools.wraps(func)
    async def _WrapErrorFuncWrapper(self, *args, error_context: _ErrorContext,
                                    **kwargs):
      try:
        return await func(self, *args, error_context=error_context, **kwargs)
      except _HTTPError as e:
        logger.exception(f'Failed to {action}. {e.reason}')
        error_id = uuid.uuid4().hex
        error_context['e.response'] = e.response
        error_context['error_id'] = error_id
        return ResponseModelT(success=None,
                              error=ErrorModelT(
                                  message=f'Failed to {action}. {e.reason}',
                                  status_code=e.status,
                                  name='HTTPError',
                                  context=error_context.UserDump(),
                                  error_id='HTTPError'))
      except Exception as e:
        logger.exception(f'Failed to {action}.')
        error_id = uuid.uuid4().hex
        error_context['e.__class__.__name__'] = e.__class__.__name__
        error_context['error_id'] = error_id
        return ResponseModelT(success=None,
                              error=ErrorModelT(
                                  message=f'Failed to {action}.',
                                  status_code=500,
                                  name='InternalError',
                                  context=error_context.UserDump(),
                                  error_id=error_id))

    return _WrapErrorFuncWrapper

  return _WrapErrorDecorator


class RemoteServer(ServerBase):
  """An implementation of ServerBase that uses a remote server (see web/routes.py)."""

  def __init__(self, *, lowda_url: str, debug_path: Path, console: Console):
    self._debug_path: Path = debug_path
    self._lowda_url = lowda_url
    self._console = console

  @_WrapError(action='upload workflow',
              ResponseModelT=ServerBase.UploadWorkflowRes,
              ErrorModelT=ServerBase.UploadWorkflowError)
  async def _UploadWorkflow(
      self, req: ServerBase.UploadWorkflowReq,
      error_context: _ErrorContext) -> ServerBase.UploadWorkflowRes:
    url = _ConstructEndpoint(lowda_url=self._lowda_url,
                             endpoint=WORKFLOW_UPLOAD_ENDPOINT)
    error_context['url'] = url
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True, by_alias=True))
    return await _ClientPostWithRequests(action='upload workflow',
                                         url=url,
                                         req=req,
                                         resp_type=ServerBase.UploadWorkflowRes,
                                         error_context=error_context,
                                         console=self._console)

  async def UploadWorkflow(
      self, req: ServerBase.UploadWorkflowReq) -> ServerBase.UploadWorkflowRes:
    error_context = await _ErrorContext.Create(
        debug_path=self._debug_path, key='RemoteServer.UploadWorkflow')
    return await self._UploadWorkflow(req, error_context)

  @_WrapError(action='download workflow',
              ResponseModelT=ServerBase.DownloadWorkflowRes,
              ErrorModelT=ServerBase.DownloadWorkflowError)
  async def _DownloadWorkflow(
      self, req: ServerBase.DownloadWorkflowReq,
      error_context: _ErrorContext) -> ServerBase.DownloadWorkflowRes:
    url = _ConstructEndpoint(lowda_url=self._lowda_url,
                             endpoint=WORKFLOW_DOWNLOAD_ENDPOINT)
    error_context['url'] = url
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True, by_alias=True))
    return await _ClientPostWithRequests(
        action='download workflow',
        url=url,
        req=req,
        resp_type=ServerBase.DownloadWorkflowRes,
        error_context=error_context,
        console=self._console)

  async def DownloadWorkflow(
      self,
      req: ServerBase.DownloadWorkflowReq) -> ServerBase.DownloadWorkflowRes:
    error_context = await _ErrorContext.Create(
        debug_path=self._debug_path, key='RemoteServer.DownloadWorkflow')
    return await self._DownloadWorkflow(req, error_context)

  @_WrapError(action='execute',
              ResponseModelT=ServerBase.ExecuteRes,
              ErrorModelT=ServerBase.ExecuteError)
  async def _Execute(self, req: ServerBase.ExecuteReq,
                     error_context: _ErrorContext) -> ServerBase.ExecuteRes:
    url = _ConstructEndpoint(lowda_url=self._lowda_url,
                             endpoint=JOB_EXECUTE_ENDPOINT)
    error_context['url'] = url
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True, by_alias=True))
    return await _ClientPostWithRequests(action='execute',
                                         url=url,
                                         req=req,
                                         resp_type=ServerBase.ExecuteRes,
                                         error_context=error_context,
                                         console=self._console)

  async def Execute(self, req: ServerBase.ExecuteReq) -> ServerBase.ExecuteRes:
    error_context = await _ErrorContext.Create(debug_path=self._debug_path,
                                               key='RemoteServer.Execute')
    return await self._Execute(req, error_context)

  async def _Touch(self, req: ServerBase.TouchReq,
                   error_context: _ErrorContext) -> ServerBase.TouchRes:

    url = _ConstructEndpoint(lowda_url=self._lowda_url,
                             endpoint=JOB_TOUCH_ENDPOINT)
    error_context['url'] = url
    error_context['req'] = await error_context.LargeToFile(
        'req', req.model_dump(mode='json', round_trip=True, by_alias=True))
    return await _ClientPostWithRequests(action='touch',
                                         url=url,
                                         req=req,
                                         resp_type=ServerBase.TouchRes,
                                         error_context=error_context.Copy(),
                                         console=self._console)

  async def Touch(self, req: ServerBase.TouchReq) -> ServerBase.TouchRes:
    error_context = await _ErrorContext.Create(debug_path=self._debug_path,
                                               key='RemoteServer.Touch')
    return await self._Touch(req, error_context)
