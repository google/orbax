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
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.export import constants
from orbax.export import obm_configs
from orbax.export.data_processors import data_processor_base


def _jax_spec_from(spec: Any) -> jax.ShapeDtypeStruct | Any:
  """Converts a ShloTensorSpec to a jax.ShapeDtypeStruct."""
  if isinstance(spec, shlo_function.ShloTensorSpec):
    if spec.dtype == shlo_function.ShloDType.bf16:
      return jax.ShapeDtypeStruct(spec.shape, jax.numpy.bfloat16)
    return jax.ShapeDtypeStruct(
        spec.shape, shlo_function.shlo_dtype_to_np_dtype(spec.dtype)
    )
  return spec


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

    jax_input_signature = jax.tree.map(_jax_spec_from, self._input_signature)

    # Construct args_spec for jax2obm.convert.
    # We assume the callable takes (params, inputs) if params is not None,
    # and just (inputs) otherwise.
    if self._params is not None:
      # We need to get specs for params.
      # Assuming self._params is a PyTree of arrays.
      params_spec = jax.tree.map(
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self._params
      )
      args_spec = (params_spec, jax_input_signature)

      ckp_path = self._options.checkpoint_path or 'processor_checkpoint'

      def _save_checkpoint(
          model_path: str,
      ) -> Mapping[str, value.ExternalValue]:
        checkpoint_path = pathlib.Path(model_path) / ckp_path / self.name
        jax2obm.save_checkpoint(self._params, checkpoint_path)

        weight_name = self._options.weights_name or f'params_{self.name}'
        return {
            weight_name: jax2obm.convert_path_to_value(
                str(pathlib.Path(ckp_path) / self.name),
                constants.ORBAX_CHECKPOINT_MIME_TYPE,
            )
        }

      self._save_fn = _save_checkpoint
    else:
      args_spec = (jax_input_signature,)

    # Instructs the runtime to only load the model parameters from the
    # checkpoint, not all keys present in the checkpoint.
    param_names_tree = jax.tree.map_with_path(
        lambda path, _: jax.tree_util.keystr(path, simple=True, separator='.'),
        self._params,
    )

    native_serialization_platforms = (
        self._options.native_serialization_platforms
    )
    if native_serialization_platforms is None:
      platforms = None
    elif isinstance(native_serialization_platforms, str):
      platforms = [native_serialization_platforms]
    else:
      platforms = native_serialization_platforms

    if platforms is not None:
      platforms = [manifest_pb2.Platform.Value(p.upper()) for p in platforms]

    # Convert the JAX function to an Orbax Model function using jax2obm,
    # making it compatible with the Orbax Export framework.
    self._obm_function = jax2obm.convert(
        fun_jax=self._processor_callable,
        args_spec=args_spec,
        kwargs_spec={},
        platforms=platforms,
        native_serialization_disabled_checks=self._options.native_serialization_disabled_checks,
        model_param_names=jax.tree.leaves(param_names_tree),
        # TODO: b/485622993 - Add other options if needed.
    )

    self._output_signature = self._obm_function.output_signature

    self._is_prepared = True

  @property
  def options(self) -> obm_configs.Jax2ObmOptions:
    """The options for jax2obm conversion."""
    return self._options
