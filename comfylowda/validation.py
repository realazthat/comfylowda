# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import json
import pprint
import textwrap
from typing import Any

from deepdiff import DeepDiff
from pydantic import BaseModel
from typing_extensions import TypeVar

from .error_utils import _YamlDump

_BaseModelT = TypeVar('_BaseModelT', bound=BaseModel)


def _CheckModelRoundTrip(*, model: _BaseModelT, t: type[_BaseModelT]) -> None:
  model_data = model.model_dump(mode='json', round_trip=True, by_alias=True)
  model2 = t.model_validate(model_data)
  if not model2 == model:
    raise Exception(f'Model=>data=>model round trip failed for {t.__name__}.')


def _CheckJSONRoundTrip(*, json_str: str, t: type[_BaseModelT]) -> None:
  json_data = json.loads(json_str)
  model = t.model_validate_json(json_str)
  model_data = model.model_dump(mode='json', round_trip=True, by_alias=True)
  if model_data != json_data:
    raise Exception(f'data=>model=>data round trip failed for {t.__name__}.')


def _DoubleValidateJSON(*, arg_name: str, json_str: str,
                        t: type[_BaseModelT]) -> _BaseModelT:
  json_data: dict[str, Any] | None = None
  model: _BaseModelT | None = None
  model_data: dict[str, Any] | None = None
  try:
    json_data = json.loads(json_str)
    model = t.model_validate_json(json_str)
    # Parse this before checking for errors, because this is used in the
    # exception handler for debugging.
    model_data = model.model_dump(mode='json', round_trip=True, by_alias=True)
    _CheckModelRoundTrip(model=model, t=t)
    # Don't do json round trip, because it fails for floating point<=>int
    # conversions.
    # _CheckJSONRoundTrip(json_str=json_str, t=t)
    t.model_validate(model_data)
    return model
  except Exception as e:
    formatted_json_str = json_str
    try:
      formatted_json_str = json.dumps(json.loads(json_str), indent=2)
      formatted_json_str = textwrap.indent(formatted_json_str, '  ')
    except json.JSONDecodeError:
      pass
    formatted_json_yaml_str: str | None = None
    try:
      formatted_json_yaml_str = _YamlDump(json.loads(json_str))
      formatted_json_yaml_str = textwrap.indent(formatted_json_yaml_str, '  ')
    except json.JSONDecodeError:
      pass
    short_msg = ('Failed to validate JSON string for argument'
                 f' {json.dumps(arg_name)} as {t.__name__}.')

    msg = short_msg
    msg += '\nJSON:\n' + formatted_json_str
    if formatted_json_yaml_str is not None:
      msg += '\nYAML:\n' + formatted_json_yaml_str
    msg += '\nDue to error:\n' + textwrap.indent(str(e), '  ')
    msg += '\nParsed model:\n' + textwrap.indent(str(model), '  ')
    # msg += '\nDumped data:\n' + textwrap.indent(pprint.pformat(model_data),

    if model_data is not None and json_data is not None:
      ddiff = DeepDiff(model_data, json_data)
      msg += '\nDeepDiff:\n' + textwrap.indent(pprint.pformat(ddiff, indent=2),
                                               '  ')

    msg += '\n' + short_msg
    raise Exception(msg) from e
