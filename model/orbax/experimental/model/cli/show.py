# Copyright 2025 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OBM CLI `show` command."""

import os

from google.protobuf import message
from orbax.experimental.model import core as obm
from orbax.experimental.model.cli import constants
from orbax.experimental.model.cli import printer
from orbax.experimental.model.core.python import file_utils
import rich
import typer
from typing_extensions import Annotated


def show(
    model_base_dir: Annotated[
        str, typer.Argument(help='Base directory of the model')
    ],
    details: Annotated[
        str,
        typer.Option(
            help=(
                'Show the detailed information about an object or a'
                ' supplemental'
            )
        ),
    ] = '',
    verbose: Annotated[
        bool,
        typer.Option(
            '--verbose',
            '-v',
            help='Show more verbose information',
        ),
    ] = False,
) -> None:
  """Shows the summary of an Orbax model."""
  rich.print('Loading Orbax model... ')
  manifest: obm.manifest_pb2.Manifest
  try:
    manifest = obm.load(model_base_dir)
  except IOError as e:
    typer.echo(f'Failed to load Orbax model: {e}')
    raise typer.Exit(1)

  rich.print(
      f'Manifest: {manifest.ByteSize()//1024} KiB,'
      f' {len(manifest.objects)} objects,'
      f' {len(manifest.supplemental_info)} supplementals'
  )

  if details:
    rich.print(f'Showing details for {details}...')
    _show_details(model_base_dir, manifest, details, verbose)
    return

  if manifest.objects:
    table = rich.table.Table('Object', 'Type', padding=(0, 1, 1, 1))
    for name, obj in sorted(manifest.objects.items()):
      table.add_row(name, printer.top_level_object(obj))
    rich.print(table)

  if manifest.supplemental_info:
    table = rich.table.Table('Supplemental', 'Type', padding=(0, 1, 1, 1))
    for name, supp in sorted(manifest.supplemental_info.items()):
      table.add_row(name, printer.unstructured_data(supp))
    rich.print(table)


def _show_details(
    model_base_dir: str,
    manifest: obm.manifest_pb2.Manifest,
    details: str,
    verbose: bool,
) -> None:
  """Shows details for an object or a supplemental."""
  if details in manifest.objects:
    _show_object_details(details, manifest.objects[details], verbose)

  if details in manifest.supplemental_info:
    _show_supplemental_details(
        model_base_dir, details, manifest.supplemental_info[details], verbose
    )

  if (
      details not in manifest.objects
      and details not in manifest.supplemental_info
  ):
    rich.print(f'Unknown object or supplemental: {details}')
  elif not verbose:
    rich.print('Re-run with `--verbose` to show more details')


def _show_object_details(
    name: str, obj: obm.manifest_pb2.TopLevelObject, verbose: bool
) -> None:
  if obj.HasField('function'):
    _show_function_details(name, obj.function, verbose)
  elif obj.HasField('value'):
    _show_value_details(name, obj.value)
  elif obj.HasField('poly_fn'):
    pass
  else:
    rich.print('Unknown object type')


def _show_function_details(
    name: str, fn: obm.manifest_pb2.Function, verbose: bool
) -> None:
  """Shows details for a StableHLO function body."""

  table = rich.table.Table('Function', name, padding=(0, 1, 1, 1))

  if fn.body.HasField('stable_hlo_body'):
    body = fn.body.stable_hlo_body
    table.add_row('Type', 'StableHLO')
    table.add_row(
        'Visibility',
        obm.manifest_pb2.Visibility.Name(fn.visibility).lower(),
    )
    table.add_row('Size', f'{len(body.stable_hlo.inlined_bytes)} bytes')
    table.add_row('Input signature', printer.obm_type(fn.signature.input))
    table.add_row('Output signature', printer.obm_type(fn.signature.output))
    table.add_row('Lowering platforms', ','.join(body.lowering_platforms))
    table.add_row(
        'Calling convention version', f'{body.calling_convention_version}'
    )
    if fn.data_names:
      if verbose:
        table.add_row(
            'Data names',
            f'{len(fn.data_names)} total: \n'
            f'{printer.pad("\n".join(fn.data_names), " - ")}',
        )
      else:
        table.add_row('Data names', f'{len(fn.data_names)} total')
    if fn.gradient_function_name:
      table.add_row('Gradient function name', f'{fn.gradient_function_name}')
    if body.supplemental_info:
      for name, supp in sorted(body.supplemental_info.items()):
        table.add_row(name, printer.unstructured_data(supp))
  elif fn.body.HasField('other'):
    table.add_row('Type', printer.unstructured_data(fn.body.other))
  else:
    table.add_row('Type', 'Unknown function')

  rich.print(table)


def _show_value_details(name: str, value: obm.manifest_pb2.Value) -> None:
  """Shows details for a value."""
  if value.HasField('external'):
    table = rich.table.Table('External value', name, padding=(0, 1, 1, 1))
    table.add_row(
        'Loader',
        obm.manifest_pb2.LoaderType.Name(value.external.loader_type),
    )
    table.add_row('Data', printer.unstructured_data(value.external.data))
    rich.print(table)
  else:
    rich.print('Unknown value type')


def _show_supplemental_details(
    model_base_dir: str,
    name: str,
    data: obm.manifest_pb2.UnstructuredData,
    verbose: bool,
) -> None:
  match data.mime_type:
    case _:
      table = rich.table.Table('Suppemental', 'Type', padding=(0, 1, 1, 1))
      table.add_row(name, printer.unstructured_data(data))
      rich.print(table)


def _read_proto_from_data(
    model_base_dir: str,
    data: obm.manifest_pb2.UnstructuredData,
    proto: message.Message,
) -> None:
  """Reads a proto from the given UnstructuredData."""
  if data.HasField('file_system_location'):
    with file_utils.open_file(
        os.path.join(model_base_dir, data.file_system_location.string_path),
        'rb',
    ) as f:
      proto.ParseFromString(f.read())
  elif data.HasField('inlined_bytes'):
    proto.ParseFromString(data.inlined_bytes)
  elif data.HasField('inlined_string'):
    proto.ParseFromString(data.inlined_string.encode('utf-8'))
  else:
    raise ValueError('Unsupported data type')
