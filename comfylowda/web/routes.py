# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

# FASTAPI routes
#
# In the routes.py file, we define the routes for the FASTAPI application.

# Path: src/routes.py

import logging

from fastapi import APIRouter, Request, Response

from ..lowda import ServerBase

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post('/upload/')
async def Upload(
    request: Request, response: Response,
    req: ServerBase.UploadWorkflowReq) -> ServerBase.UploadWorkflowRes:
  server: ServerBase = request.app.state.server
  res = await server.UploadWorkflow(req)
  if res.error is not None:
    status_code = 500
    if res.error.status_code is not None:
      status_code = res.error.status_code
    response.status_code = status_code
  return res


@router.post('/execute')
async def Execute(request: Request, response: Response,
                  req: ServerBase.ExecuteReq) -> ServerBase.ExecuteRes:
  server: ServerBase = request.app.state.server
  res = await server.Execute(req)
  if res.error is not None:
    status_code = 500
    if res.error.status_code is not None:
      status_code = res.error.status_code
    response.status_code = status_code
  return res


@router.post('/touch')
async def Touch(request: Request,
                req: ServerBase.TouchReq) -> ServerBase.TouchRes:
  server: ServerBase = request.app.state.server
  return await server.Touch(req)
