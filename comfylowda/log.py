# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import json
import logging
import sys
import traceback
from abc import ABC, abstractmethod

from anyio import Path

from .args import CommonArgs
from .error_utils import _YamlDump


class _CustomJSONEncoder(json.JSONEncoder):

  def default(self, obj):
    try:
      return super().default(obj)
    except TypeError:
      return str(obj)


class _CustomFormatter(logging.Formatter, ABC):

  def format_exception(self, exc_info):
    return ''.join(traceback.format_exception(*exc_info))

  @abstractmethod
  def DumpRecordDict(self, record_dict: dict) -> str:
    raise NotImplementedError()

  def format(self, record):
    record_dict = {}
    record_dict['message'] = record.getMessage()

    record_dict.update(record.__dict__)
    # record.message = record.getMessage()
    # if self.usesTime():
    #   record.asctime = self.formatTime(record, self.datefmt)
    if self.usesTime():
      record_dict['asctime'] = self.formatTime(record, self.datefmt)
    record_dict.pop('exc_info', None)
    record_dict.pop('args', None)
    record_dict.pop('msg', None)

    if record.exc_info is not None:
      record_dict['exc_info'] = self.format_exception(record.exc_info)
      record_dict['exc_text'] = super().formatException(record.exc_info)

    return self.DumpRecordDict(record_dict)


class _YAMLFormatter(_CustomFormatter):

  def DumpRecordDict(self, record_dict: dict) -> str:
    return '---\n' + _YamlDump(record_dict)


class _JSONFormatter(_CustomFormatter):

  def DumpRecordDict(self, record_dict: dict) -> str:
    return json.dumps(record_dict, cls=_CustomJSONEncoder)


async def _SetupLogging(args: CommonArgs) -> None:

  handlers: list[logging.Handler] = []
  if args.json_log_path is not None:
    json_log_path = await Path(args.json_log_path).absolute()
    await json_log_path.parent.mkdir(parents=True, exist_ok=True)
    json_file_handler = logging.FileHandler(str(json_log_path))
    json_file_handler.setFormatter(_JSONFormatter())
    handlers.append(json_file_handler)
  if args.yaml_log_path is not None:
    yaml_log_path = await Path(args.yaml_log_path).absolute()
    await yaml_log_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_file_handler = logging.FileHandler(str(yaml_log_path))
    yaml_file_handler.setFormatter(_YAMLFormatter())
    handlers.append(yaml_file_handler)

  if args.log_to_stderr:
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_YAMLFormatter())
    handlers.append(stderr_handler)
  logging.basicConfig(handlers=handlers, level=args.log_level)
  logging.captureWarnings(True)

  # Make sure all loggers goes to our handlers

  for name in logging.root.manager.loggerDict:
    logger = logging.getLogger(name)
    logger.handlers = handlers
    logger.propagate = False
    logger.setLevel(args.log_level)
    logger.disabled = False
