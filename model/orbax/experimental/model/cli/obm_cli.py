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

"""Command line interface for Orbax Model.

obm_cli is a command line interface for Orbax Model. It allows users to inspect
an exported Orbax Model.
"""

from collections.abc import Sequence
import sys

from absl import app
from orbax.experimental.model.cli import constants
from orbax.experimental.model.cli import show
import rich
import typer


cli = typer.Typer(pretty_exceptions_enable=False)
cli.command(name='show')(show.show)


@cli.command()
def version() -> None:
  """Prints the OBM CLI version."""
  rich.print(f'OBM CLI version: {constants.CLI_VERSION}')


def main(unused_argv: Sequence[str]) -> None:
  cli()


if __name__ == '__main__':
  app.run(main, argv=sys.argv[:1])
