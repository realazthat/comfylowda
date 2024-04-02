# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
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

WORKFLOW_UPLOAD_ENDPOINT = '/workflow/upload/'
WORKFLOW_DOWNLOAD_ENDPOINT = '/workflow/download/'
JOB_EXECUTE_ENDPOINT = '/job/execute/'
JOB_TOUCH_ENDPOINT = '/job/touch/'


@router.post(WORKFLOW_UPLOAD_ENDPOINT)
async def WorkflowUpload(
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


@router.post(WORKFLOW_DOWNLOAD_ENDPOINT)
async def WorkflowDownload(
    request: Request, response: Response,
    req: ServerBase.DownloadWorkflowReq) -> ServerBase.DownloadWorkflowRes:
  server: ServerBase = request.app.state.server
  res = await server.DownloadWorkflow(req)
  if res.error is not None:
    status_code = 500
    if res.error.status_code is not None:
      status_code = res.error.status_code
    response.status_code = status_code
  return res


@router.post(JOB_EXECUTE_ENDPOINT)
async def JobExecute(request: Request, response: Response,
                     req: ServerBase.ExecuteReq) -> ServerBase.ExecuteRes:
  server: ServerBase = request.app.state.server
  res = await server.Execute(req)
  if res.error is not None:
    status_code = 500
    if res.error.status_code is not None:
      status_code = res.error.status_code
    response.status_code = status_code
  return res


@router.post(JOB_TOUCH_ENDPOINT)
async def JobTouch(request: Request,
                   req: ServerBase.TouchReq) -> ServerBase.TouchRes:
  server: ServerBase = request.app.state.server
  return await server.Touch(req)
