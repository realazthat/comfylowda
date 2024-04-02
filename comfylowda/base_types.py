# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

from typing import Generic, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypeVar

################################################################################
_SuccessT = TypeVar('_SuccessT', bound=BaseModel)
_ErrorT = TypeVar('_ErrorT', bound=BaseModel)


class Response(BaseModel, Generic[_SuccessT, _ErrorT]):
  success: Optional[_SuccessT]
  error: Optional[_ErrorT]


class ResponseErrorBase(BaseModel):
  error_id: str = Field(..., description='The error id.')
  status_code: Optional[int] = Field(
      ..., description='The error status code. To be used in HTTP responses.')
  name: str = Field(..., description='The error name.')
  message: str = Field(..., description='The error message.')
  context: dict = Field(..., description='The error context.')


################################################################################
