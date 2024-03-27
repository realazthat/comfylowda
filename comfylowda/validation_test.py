# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import unittest

from anyio import Path

from .comfy_schema import Workflow
from .lowda_types import FileDownloadMapSpec, OutputMapping, OutputPPKind
from .validation import _DoubleValidateJSON


class Tests(unittest.IsolatedAsyncioTestCase):

  async def test_ValidateOutputMapping(self):
    om = OutputMapping(
        field='images[0]',
        name='Preview Image',
        node='25',
        pp=OutputPPKind.FILE,
        spec=FileDownloadMapSpec(mode='TRIPLET', pfx=None),
        user_json_spec='ANY',
        user_value=None,
    )
    _DoubleValidateJSON(arg_name='-om',
                        json_str=om.model_dump_json(round_trip=True,
                                                    by_alias=True),
                        t=OutputMapping)

  async def test_ValidateWorkflow(self):
    path = await Path('comfylowda/assets/sdxlturbo_example.json').absolute()
    _DoubleValidateJSON(arg_name='-workflow',
                        json_str=await path.read_text(),
                        t=Workflow)


if __name__ == '__main__':
  unittest.main()
