# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

from typing import List, NamedTuple

from pydantic import BaseModel, ConfigDict, Field


class WorkflowGroupBounding(NamedTuple):
  x: float
  y: float
  width: float
  height: float


class WorkflowGroup(BaseModel):
  title: str
  bounding: WorkflowGroupBounding
  color: str
  font_size: float
  locked: bool


class WorkflowNodePosition(NamedTuple):
  x: float
  y: float


class WorkflowNodeSize(BaseModel):
  model_config = ConfigDict(populate_by_name=True)

  width: float = Field(..., alias='0')
  height: float = Field(..., alias='1')


class WorkflowNodeInput(BaseModel):
  name: str
  type: str
  link: int


class WorkflowNodeOutput(BaseModel):
  name: str
  type: str
  links: List[int] | None
  slot_index: int | None = None


class WorkflowNode(BaseModel):
  id: int
  type: str
  pos: WorkflowNodePosition
  size: WorkflowNodeSize
  flags: dict
  order: int
  mode: int
  inputs: List[WorkflowNodeInput] = Field(default_factory=list)
  outputs: List[WorkflowNodeOutput] = Field(default_factory=list)
  properties: dict
  widgets_values: list = Field(default_factory=list)


class Workflow(BaseModel):
  last_node_id: int
  last_link_id: int
  nodes: List[WorkflowNode]
  # TODO: Make this list more specific.
  links: list
  groups: List[WorkflowGroup]
  config: dict
  extra: dict
  version: float
