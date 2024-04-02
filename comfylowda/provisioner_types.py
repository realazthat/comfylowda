# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

from pydantic import BaseModel

from .base_types import Response, ResponseErrorBase
from .lowda_types import ComfyRemoteInfo, ProvisioningSpec


class ProvisionReq(BaseModel):
  id: str
  bundle: ProvisioningSpec
  keepalive: float


class ProvisionSuccess(BaseModel):
  id: str
  comfy_info: ComfyRemoteInfo


class ProvisionError(ResponseErrorBase):
  pass


class ProvisionRes(Response[ProvisionSuccess, ProvisionError]):
  pass


class TouchReq(BaseModel):
  id: str
  keepalive: float


class TouchSuccess(BaseModel):
  pass


class TouchError(ResponseErrorBase):
  pass


class TouchRes(Response[TouchSuccess, TouchError]):
  pass
