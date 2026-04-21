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

"""The data processor module for JAX-based pre/post-processors."""

from collections.abc import Callable, Mapping, Set
import pathlib
from typing import Any

import jax
import jaxtyping
from orbax.export import obm_configs
from orbax.export.data_processors import data_processor_base


class JaxDataProcessor(data_processor_base.DataProcessor):
  """A DataProcessor for JAX-based processors.

  The properties of this class are available only after `prepare()` is called.
  If a property is accessed before `prepare()` is called, it will raise a
  RuntimeError.
  """

  def __init__(
      self,
      processor_callable: Callable[..., Any],
      name: str = '',
      *,
      input_keys: Set[str] = frozenset(),
      output_keys: Set[str] = frozenset(),
      params: Any = None,
      options: obm_configs.Jax2ObmOptions | None = None,
  ):
    """Initializes the instance.

    Args:
      processor_callable: A Python callable.
      name: The name of data processor.
      input_keys: The input keys of the data processor.
      output_keys: The output keys of the data processor.
      params: The parameters for the processor (optional).
      options: Options for jax2obm conversion (optional).
    """
    super().__init__(name=name, input_keys=input_keys, output_keys=output_keys)
    self._processor_callable = processor_callable
    self._params = params
    self._options = obm_configs.Jax2ObmOptions() if options is None else options
    self._is_prepared = False

  def prepare(
      self,
      available_tensor_specs: jaxtyping.PyTree,
  ) -> None:
    """Prepares the data processor for export.

    This method transforms `processor_callable` to an OBM (Orbax Model)
    function using `jax2obm.convert`.

    Args:
      available_tensor_specs: The input signature for tracing the JAX function.

    Raises:
      RuntimeError: If `prepare()` is called more than once.
    """
    if self._is_prepared:
      raise RuntimeError('`prepare()` can only be called once.')

    if not self._name:
      raise ValueError(
          'JaxDataProcessor.name must not be empty when calling `prepare()`.'
      )
    if not self.input_keys:
      input_signature = available_tensor_specs
    elif not isinstance(available_tensor_specs, Mapping):
      raise ValueError(
          '`available_tensor_specs` must be a Mapping if `input_keys` are'
          ' provided.'
      )
    else:
      missing_keys = self.input_keys - available_tensor_specs.keys()
      if missing_keys:
        raise ValueError(
            f'Input keys {missing_keys!r} not found in'
            f' `available_tensor_specs`: {available_tensor_specs.keys()!r}'
        )
      input_signature = {k: available_tensor_specs[k] for k in self.input_keys}

    self._input_signature = input_signature

    # Construct args_spec for jax2obm.convert.
    # We assume the callable takes (params, inputs) if params is not None,
    # and just (inputs) otherwise.
    if self._params is not None:
      # We need to get specs for params.
      # Assuming self._params is a PyTree of arrays.
      params_spec = jax.tree.map(
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self._params
      )
      args_spec = (params_spec, self._input_signature)

      def _save_supplemental(model_path: str) -> Mapping[str, Any]:
        checkpoint_path = pathlib.Path(model_path) / 'processor' / self.name
        jax2obm.save_checkpoint(self._params, checkpoint_path)
        return {}

      self._save_fn = _save_supplemental
    else:
      args_spec = (self._input_signature,)

    # Convert the JAX function to an Orbax Model function using jax2obm,
    # making it compatible with the Orbax Export framework.
    self._obm_function = jax2obm.convert(
        fun_jax=self._processor_callable,
        args_spec=args_spec,
        kwargs_spec={},
        native_serialization_disabled_checks=self._options.native_serialization_disabled_checks,
        # TODO: b/485622993 - Add other options if needed.
    )

    self._output_signature = self._obm_function.output_signature

    self._is_prepared = True
