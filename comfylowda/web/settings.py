# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

from typing import List

from pydantic import BaseModel

from .. import _build_version


class AppSettings(BaseModel):
  project_name: str = 'Comfylowda'
  version: str = _build_version
  description: str = 'Comfy Catapult Lowda'
  docs_url: str = '/docs'
  redoc_url: str = '/redoc'
  debug: bool = False
  hostname: str
  port: int
  allowed_hosts: List[str] = ['*']
