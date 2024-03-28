# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The ComfyLowda project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import argparse
import collections
import collections.abc
import enum
import functools
import json
import sys
import textwrap
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined, PydanticUndefinedType
from slugify import slugify
from typing_extensions import Literal, TypedDict, TypeVar, get_args, get_origin

AppendOrNargs = Literal['REPEAT', 'NARGS']
"""How to interpret a list. Either allow the --arg to repeat or just allow the value to repeat."""
AppendOrNargsValidator = TypeAdapter[AppendOrNargs](AppendOrNargs)

# TODO: Add CHOICE and MUTEX and GROUP?
# TODO: Make COMMAND optionally mandatory.
ParseAs = Literal['ENUMERATE', 'COMMAND', 'STRING']
"""How to interpret a BaseModel.

ENUMERATE: The BaseModel values are individually listed as arguments.
COMMAND: The BaseModel values are subcommands. They must all be optional.
STRING: The BaseModel must be encoded, as JSON or YAML. Use
  `ArgConfigDict.parse_as` to specify the encoding format.
"""
ParseAsValidator = TypeAdapter[ParseAs](ParseAs)

WireFormat = Literal['JSON', 'YAML']
"""What kind of string the BaseModel is encoded as, for ParseAs=STRING"""
WireFormatValidator = TypeAdapter[WireFormat](WireFormat)


class ArgConfigDict(TypedDict, total=False):
  append_or_nargs: AppendOrNargs
  """How to interpret a list."""
  parse_as: ParseAs
  """How to interpret a BaseModel."""
  wire_format: WireFormat
  """What kind of string the BaseModel is encoded as, for ParseAs=STRING"""


def _IsTypeA(inner_type: Any, types: Union[Any, Tuple[Any, ...]]) -> bool:
  """_summary_

  Copied from pydantic2_argparse/utils/types.py: is_field_a().
  """
  if not isinstance(types, tuple):
    types = (types, )

  inner_type = get_origin(inner_type) or inner_type
  is_valid = all(isinstance(t, type) for t in (*types, inner_type))
  return (inner_type in types or (is_valid and isinstance(inner_type, types))
          or (is_valid and issubclass(inner_type, types)))


def _IsContainer(underlying_type: Optional[type[Any]]) -> bool:
  if _IsTypeA(underlying_type,
              (collections.abc.Mapping, enum.Enum, str, bytes)):
    return False
  return _IsTypeA(underlying_type, (collections.abc.Container))


ParsableType = Union[str, int, float, bool, Path]


def _IsOptional(t: Optional[type[Any]]) -> bool:
  origin_type = get_origin(t)
  if not (_IsTypeA(origin_type,
                   (type[Union], Union, UnionType)) or origin_type == Union):
    return False
  union_args = get_args(t)
  if len(union_args) == 2 and type(None) in union_args:
    return True
  return type(None) in union_args


def _NonOptionalType(t: type[Any] | Type[Any]) -> type[Any] | Type[Any]:
  origin_type = get_origin(t)
  if origin_type is None or not _IsOptional(t):
    raise ValueError(
        f'Expected get_origin(t)=Union, got {origin_type}, t={t}. English: This should only be used on an Optional type.'
    )

  union_args = get_args(t)

  union_args = [a for a in union_args if a is not type(None)]
  if len(union_args) != 1:
    raise ValueError(
        f'Expected exactly one non-None type in {union_args}, got {len(union_args)}, with t={t}, get_origin(t)={origin_type}. English: This should only be used on an Optional type.'
    )
  return union_args[0]


_BaseModelT = TypeVar('_BaseModelT', bound=BaseModel)


class ParserHelper:

  def __init__(self):
    self._dest2type: Dict[str, type[Any] | None] = {}
    self._id2type: Dict[str, type[Any] | None] = {}
    self._id2dest: Dict[str, str] = {}
    self._id2parent_dest: Dict[str, str | None] = {}
    self._dest2id: Dict[str, str] = {}
    self.verbose = False
    self.print = functools.partial(print, file=sys.stderr)

  class Context(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    append_or_nargs: AppendOrNargs
    parse_as: ParseAs
    wire_format: WireFormat
    subparsers: argparse._SubParsersAction | None

  def _MakeDefaultContext(self) -> Context:
    return self.Context(append_or_nargs='REPEAT',
                        parse_as='STRING',
                        wire_format='JSON',
                        subparsers=None)

  def _CombineContext(self, context: Context, t: type[BaseModel]) -> Context:
    if t is not None and (t is BaseModel or issubclass(t, BaseModel)):
      arg_config: ArgConfigDict = getattr(t, 'arg_config', ArgConfigDict())
      parse_as = ParseAsValidator.validate_python(
          arg_config.get('parse_as', 'ENUMERATE'))
      append_or_nargs = AppendOrNargsValidator.validate_python(
          arg_config.get('append_or_nargs', context.append_or_nargs))
      wire_format = WireFormatValidator.validate_python(
          arg_config.get('wire_format', context.wire_format))
      return self.Context(append_or_nargs=append_or_nargs,
                          parse_as=parse_as,
                          wire_format=wire_format,
                          subparsers=None)
    return context

  def _GetValidatorAsType(self, user_id: str, t: Any,
                          context: Context) -> Callable[[str], Any]:
    if _IsOptional(t):
      t = _NonOptionalType(t)

    if _IsContainer(t):
      contained_types = get_args(t)
      if not len(contained_types) == 1:
        raise ValueError(
            f'Expected exactly one contained type, got {contained_types}')

      contained_type = contained_types[0]
      return self._GetValidatorAsType(user_id, contained_type, context=context)

    # Note this must go first, i think int is qualifying as a descendant of bool
    # or something.
    if _IsTypeA(t, (bool, )):
      raise ValueError(
          'Um weird, we should be using the _BoolAction to validate a bool')
    if _IsTypeA(t, (str, int, float, Path)):
      return t
    if _IsTypeA(t, BaseModel):
      if (not isinstance(t, type) or not issubclass(t, BaseModel)):
        raise ValueError(f'Expected a BaseModel type, got {t}')

      context = self._CombineContext(context, t)
      if context.parse_as == 'STRING':

        def model_validate_string(value: str):
          try:
            if context.wire_format == 'JSON':
              return t.model_validate_json(value)
            elif context.wire_format == 'YAML':
              value_py = yaml.safe_load(value)
              return t.model_validate_json(value_py)
            else:
              raise ValueError(f'Unsupported wire_format {context.wire_format}')
          except Exception as e:
            raise argparse.ArgumentError(
                None,
                f'For argument {user_id}: Failed to parse {context.wire_format} as {t.__name__}:'
                f"\n{textwrap.indent(str(e), '  ')}"
                f"\nArgument value:\n{textwrap.indent(value, '  ')}") from e

        return model_validate_string
      elif context.parse_as == 'ENUMERATE':
        raise ValueError(
            f'Cannot directly validate a BaseModel with parse_as ENUMERATE, got {t}, context={context}. In english: This should not have been called directly on the base model!'
        )
      elif context.parse_as == 'COMMAND':
        raise ValueError(
            f'Cannot directly validate a BaseModel with parse_as COMMAND, got {t}, context={context}. In english: This should not have been called directly on the base model!'
        )
    raise ValueError(f'Unsupported type {t}')

  def _GetFieldNameOrFlags(self, field_name: str, field: FieldInfo,
                           context: Context) -> str:
    # TODO: Make it possible to turn off root-based names.
    # field_name_or_flag_root: str = f'{name_or_flags}.'
    name_root: str = ''
    field_name_or_flags: str
    field_name_or_flags = f'{slugify(field_name)}'
    if field.alias:
      field_name_or_flags = f'{slugify(field.alias)}'
    # TODO: Add other ways to choose the field_name, e.g from alias, or from metadata.

    # TODO: Make it possible to turn off root-based names.
    field_name_or_flags = f'{name_root}{field_name_or_flags}'

    child_underlying_type = field.annotation
    if child_underlying_type is not None and _IsOptional(child_underlying_type):
      child_underlying_type = _NonOptionalType(child_underlying_type)

    if _IsTypeA(child_underlying_type, (BaseModel, )):
      if (not isinstance(child_underlying_type, type)
          or not issubclass(child_underlying_type, BaseModel)):
        raise ValueError(
            f'Expected a BaseModel type, got {child_underlying_type}')
      bm_type: type[BaseModel] = child_underlying_type
      children_context = self._CombineContext(context, bm_type)
      if children_context.parse_as == 'COMMAND':
        return field_name_or_flags

    if _IsContainer(child_underlying_type):
      field_name_or_flags = f'-{field_name_or_flags}'
    else:
      # TODO: Make it possible to make this positional.
      field_name_or_flags = f'--{field_name_or_flags}'
    return field_name_or_flags

  def _AddType(self, *, parser: argparse.ArgumentParser, id: str,
               parent_dest: str | None, name_or_flags: str, metavar: str,
               t: type[Any] | Type[Any], description: str | None,
               is_required: bool, default: PydanticUndefinedType | Any | None,
               context: Context, add_args_kwargs: Dict[str, Any] | None,
               error_context: Dict[str, Any]):
    if add_args_kwargs is None:
      add_args_kwargs = {}

    has_default = default != PydanticUndefined
    is_optional = _IsOptional(t)
    underlying_type: type[Any] | Type[Any] = t

    if is_optional:
      underlying_type = _NonOptionalType(t)

    if is_required and is_optional:
      # Q: How can this be both optional and required?
      # A: Because it can be optional in the sense that it can be None, but
      # required in the sense that it must be present.
      pass

    dest = slugify(id).replace('-', '_')
    if id in self._id2dest:
      raise ValueError(f'Duplicate id {json.dumps(id)}')
    if dest in self._dest2id:
      raise ValueError(f'Duplicate dest {json.dumps(dest)}')
    self._id2dest[id] = dest
    self._dest2id[dest] = id
    self._id2parent_dest[id] = parent_dest

    if self.verbose:
      self.print()
      self.print('id:', id, file=sys.stderr)
      self.print('name_or_flags:', name_or_flags)
      self.print('t:', t)
      self.print('has_default:', has_default)
      self.print('is_optional:', is_optional)
      self.print('is_required:', is_required)
      self.print('dest:', dest)
      self.print('default:', default)
      self.print('type(default):', type(default))
      self.print('type(default) is not PydanticUndefinedType:',
                 type(default) is not PydanticUndefinedType)
      self.print('type(default) == PydanticUndefinedType:',
                 type(default) == PydanticUndefinedType)
      self.print('default is PydanticUndefined:', default is PydanticUndefined)
      self.print('default == PydanticUndefined:', default == PydanticUndefined)
      self.print('isinstance(default, PydanticUndefinedType):',
                 isinstance(default, PydanticUndefinedType))
      self.print('self._id2dest:', self._id2dest)
      self.print('t:', t)
      self.print('isinstance(t, (str, int, float, bool, Path)):',
                 _IsTypeA(t, bool))
      self.print('add_args_kwargs:', add_args_kwargs)

    # TODO: Need to test for/unwrap annotated.
    if _IsTypeA(underlying_type, (Literal, )):
      help_desc = description or ''
      # Note there is no single type that can be gleaned from Literal/choices,
      # because choices and Literal can contain any type that argparse can
      # handle.
      # TODO: Check the types inside, to make sure that they are parsable?

      help_desc += f' (default: {default})' if not is_required else ' (required)'
      kwargs = {}
      if not isinstance(default, PydanticUndefinedType):
        kwargs['default'] = default
      parser.add_argument(name_or_flags,
                          metavar=metavar,
                          dest=dest,
                          choices=get_args(underlying_type),
                          required=is_required,
                          help=help_desc,
                          **kwargs,
                          **add_args_kwargs)
    elif _IsTypeA(underlying_type, (bool, )):
      if (not isinstance(underlying_type, type)
          or not issubclass(underlying_type, bool)):
        raise ValueError(f'Expected a bool type, got {underlying_type}')
      if (not isinstance(t, type)):
        raise ValueError(f'Expected a type, got {t}')
      help_desc = description or ''
      help_desc += ' 0 or 1'
      if has_default and isinstance(default, bool):
        default = '1' if default else '0'
      if has_default:
        help_desc += f' (default: {default})'
      if is_required:
        help_desc += ' (required)'

      class _BoolAction(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
          values = True if values == '1' else False
          setattr(namespace, self.dest, values)

      kwargs = {}
      if has_default:
        kwargs['default'] = default
      parser.add_argument(name_or_flags,
                          metavar=metavar,
                          dest=dest,
                          action=_BoolAction,
                          choices=['0', '1'],
                          required=is_required,
                          help=help_desc,
                          **kwargs,
                          **add_args_kwargs)
    elif _IsTypeA(underlying_type, (str, int, float, bool, Path)):
      if (not isinstance(underlying_type, type)
          or not issubclass(underlying_type, (str, int, float, bool, Path))):
        raise ValueError(
            f'Expected a str, int, float, bool, or Path type, got {underlying_type}'
        )
      readable_type: str = underlying_type.__name__

      help_desc = description or ''
      help_desc += f' ({readable_type})'
      if has_default:
        help_desc += f' (default: {default})'
      if is_required:
        help_desc += ' (required)'

      kwargs = {}
      if has_default:
        kwargs['default'] = default
      parser.add_argument(name_or_flags,
                          metavar=metavar,
                          dest=dest,
                          type=self._GetValidatorAsType(name_or_flags,
                                                        t,
                                                        context=context),
                          required=is_required,
                          help=help_desc,
                          **kwargs,
                          **add_args_kwargs)
    elif _IsContainer(underlying_type):

      contained_type = get_args(underlying_type)[0]
      is_required = False
      help_desc = description or ''
      help_desc += f' ({contained_type.__name__})'
      if context.append_or_nargs == 'REPEAT':
        help_desc += ' (Can be repeated)'
      if has_default:
        help_desc += f' (default: {default})'
      if is_required:
        help_desc += ' (required)'
      kwargs = {}
      if has_default:
        kwargs['default'] = default
      parser.add_argument(name_or_flags,
                          metavar=metavar,
                          dest=dest,
                          action='append',
                          type=self._GetValidatorAsType(name_or_flags,
                                                        t,
                                                        context=context),
                          required=is_required,
                          help=help_desc,
                          **kwargs,
                          **add_args_kwargs)
    elif _IsTypeA(underlying_type, BaseModel):
      if (not isinstance(underlying_type, type)
          or not issubclass(underlying_type, BaseModel)):
        raise ValueError(f'Expected a BaseModel type, got {underlying_type}')
      bm_type: type[BaseModel] = underlying_type

      children_context = self._CombineContext(context, bm_type)
      if children_context.parse_as == 'STRING':
        help_desc = description or ''
        # TODO: Show example.
        help_desc += f' (Type {children_context.wire_format} string, in the form of {bm_type.__name__})'
        # TODO: Format default better, especially if its a BaseModel.
        if has_default:
          help_desc += f' (default: {default})'
        if is_required:
          help_desc += ' (required)'
        kwargs = {}
        if has_default:
          kwargs['default'] = default
        parser.add_argument(name_or_flags,
                            metavar=metavar,
                            dest=dest,
                            type=self._GetValidatorAsType(
                                name_or_flags, t, context=children_context),
                            required=is_required,
                            help=help_desc,
                            **kwargs,
                            **add_args_kwargs)
      elif children_context.parse_as == 'ENUMERATE':

        field: FieldInfo
        for field_name, field in bm_type.model_fields.items():
          field_annotation = field.annotation
          if field_annotation is None:
            raise ValueError(
                f'field {json.dumps(field_name)} at {json.dumps(id)} has a annotation of None'
            )

          self._AddType(parser=parser,
                        id=f'{id}.{slugify(field_name)}',
                        parent_dest=dest,
                        name_or_flags=self._GetFieldNameOrFlags(
                            field_name, field, context=children_context),
                        metavar=field_name.upper(),
                        t=field_annotation,
                        description=field.description,
                        is_required=field.is_required(),
                        default=field.get_default(call_default_factory=True),
                        context=children_context,
                        add_args_kwargs=add_args_kwargs,
                        error_context={'parent_error_context': error_context})
      elif children_context.parse_as == 'COMMAND':
        if not is_optional:
          raise ValueError(
              f'Expected a COMMAND type to be optional, got {t}, id={id}')

        subparsers_dest = f'{parent_dest}_command'
        if context.subparsers is None:
          if parent_dest is None:
            raise ValueError(
                f'Adding a COMMAND type requires a parent_dest, got {parent_dest}'
            )

          context.subparsers = parser.add_subparsers(title='commands',
                                                     dest=subparsers_dest)

        command: argparse.ArgumentParser = context.subparsers.add_parser(
            name_or_flags,
            description=description,
            help=description,
            formatter_class=parser.formatter_class)
        field: FieldInfo
        for field_name, field in bm_type.model_fields.items():
          field_annotation = field.annotation
          if field_annotation is None:
            raise ValueError(
                f'field {json.dumps(field_name)} at {json.dumps(id)} has a annotation of None'
            )
          self._AddType(parser=command,
                        id=f'{id}.{slugify(field_name)}',
                        parent_dest=dest,
                        name_or_flags=self._GetFieldNameOrFlags(
                            field_name, field, context=children_context),
                        metavar=field_name.upper(),
                        t=field_annotation,
                        description=field.description,
                        is_required=field.is_required(),
                        default=field.get_default(call_default_factory=True),
                        context=children_context,
                        add_args_kwargs=add_args_kwargs,
                        error_context={'parent_error_context': error_context})

      else:
        raise ValueError(
            f'Unsupported parse_as {children_context.parse_as}, id={id}')
    else:
      raise ValueError(
          f'Unsupported type {t}, underlying_type={underlying_type}, id={id}')

  def _ParseAny(self, *, args: argparse.Namespace, id: str,
                field_name: str | None, t: type[Any] | Type[Any],
                context: Context) -> Any:

    underlying_type = t
    if _IsOptional(underlying_type):
      underlying_type = _NonOptionalType(underlying_type)
    # In some cases, like a subparser, the dest is not filled in at all.
    #
    # Instead the add_subparsers() dest is filled in instead, with the name of
    # the chosen command.
    #
    # So in that case, we return None. The subparser/'COMMAND' is forced to be
    # optional for this reason.
    # if not hasattr(args, self._id2dest[id]):
    #   return None

    # value = getattr(args, self._id2dest[id], None)
    # if value is None:
    #   return None

    if _IsTypeA(underlying_type, (str, int, float, bool, Path)):
      value = getattr(args, self._id2dest[id], None)
      if not _IsTypeA(type(value), underlying_type):
        raise ValueError(f'Expected {underlying_type}, got {type(value)}')
      return getattr(args, self._id2dest[id])
    elif _IsContainer(underlying_type):
      # contained_type = typing.get_args(underlying_type)[0]
      arg_values = getattr(args, self._id2dest[id])
      if arg_values is None:
        return None
      return [v for v in arg_values]
    elif _IsTypeA(underlying_type, BaseModel):
      if (not isinstance(underlying_type, type)
          or not issubclass(underlying_type, BaseModel)):
        raise ValueError(f'Expected a BaseModel type, got {underlying_type}')
      bm_type: type[BaseModel] = underlying_type
      context = self._CombineContext(context, bm_type)
      if context.parse_as == 'STRING':
        return self._ParseStringBaseModel(args=args,
                                          id=id,
                                          t=bm_type,
                                          context=context)
      elif context.parse_as == 'ENUMERATE':
        return self._ParseEnumeratedBaseModel(args=args,
                                              id=id,
                                              t=bm_type,
                                              context=context)
      elif context.parse_as == 'COMMAND':
        parent_dest = self._id2parent_dest[id]
        subparsers_dest = f'{parent_dest}_command'
        command_name = getattr(args, subparsers_dest)
        if command_name != field_name:
          return None
        return self._ParseEnumeratedBaseModel(args=args,
                                              id=id,
                                              t=bm_type,
                                              context=context)
      else:
        raise ValueError(f'Unsupported parse_as {context.parse_as}')
    else:
      return getattr(args, self._id2dest[id], None)

  def _ParseStringBaseModel(self, *, args: argparse.Namespace, id: str,
                            t: type[_BaseModelT],
                            context: Context) -> _BaseModelT:
    value = getattr(args, self._id2dest[id])
    if not _IsTypeA(value, _BaseModelT):
      raise ValueError(f'Expected {t}, got {type(value)}')
    return value

  def _ParseEnumeratedBaseModel(self, *, args: argparse.Namespace, id: str,
                                t: type[_BaseModelT],
                                context: Context) -> _BaseModelT:
    value_dict = {}
    for field_name, field in t.model_fields.items():
      child_id = f'{id}.{slugify(field_name)}'
      field_annotation = field.annotation
      if field_annotation is None:
        raise ValueError(
            f'field {json.dumps(field_name)} at {json.dumps(id)} has a annotation of None'
        )
      value_dict[field_name] = self._ParseAny(args=args,
                                              id=child_id,
                                              field_name=field_name,
                                              t=field_annotation,
                                              context=context)
    return t.model_validate(value_dict)

  def Parse(self, args: argparse.Namespace, t: type[_BaseModelT],
            context: Context) -> _BaseModelT:
    return self._ParseEnumeratedBaseModel(args=args,
                                          id='root',
                                          t=t,
                                          context=context)
