# Copyright 2026 The Orbax Authors.
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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PluginExecutable is a class for executing plugin programs over the IFRT Proxy."""

from collections.abc import Sequence
import concurrent.futures
import threading
import jax
from jax.extend import ifrt_programs
from jax.interpreters import pxla


ifrt_programs = ifrt_programs.ifrt_programs


class PluginExecutable:
  """Class for running compiled IFRT program over the IFRT Proxy."""

  def __init__(self, prog_str: str):
    ifrt_client = jax.local_devices()[0].client
    program = ifrt_programs.make_plugin_program(prog_str)
    options = ifrt_programs.make_plugin_compile_options()
    self.compiled = ifrt_client.compile_ifrt_program(program, options)

  def call(
      self,
      in_arr: Sequence[jax.Array | Sequence[jax.Array]] = (),
      out_shardings: Sequence[jax.sharding.Sharding] = (),
      out_avals: Sequence[jax.core.ShapedArray] = (),
      out_committed: bool = True,
  ) -> tuple[Sequence[jax.Array], concurrent.futures.Future[None]]:
    """Runs the compiled IFRT program and returns the result and a future."""
    results_with_token = self.compiled.execute_sharded(in_arr, with_tokens=True)

    out_arr = results_with_token.consume_with_handlers(
        pxla.global_avals_to_results_handler(
            out_avals, out_shardings, out_committed
        ).handlers
    )

    out_fut = concurrent.futures.Future()

    def call_on_done():
      try:
        results_with_token.consume_token().block_until_ready()
      except Exception as e:  # pylint: disable=broad-exception-caught
        out_fut.set_exception(e)
        return
      out_fut.set_result(None)

    t = threading.Thread(
        target=call_on_done, name="plugin_executable_call_on_done"
    )
    t.start()

    return (out_arr, out_fut)
