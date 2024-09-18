# Copyright 2024 The Orbax Authors.
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

"""Wraps JAX functions and parameters into a tf.Module."""

from collections.abc import Callable, Mapping
from typing import Any, Union

from jax import export as jax_export
from orbax.export.modules import orbax_module_base
from orbax.export.typing import ApplyFn
from orbax.export.typing import PyTree


class ObmModule(orbax_module_base.OrbaxModuleBase):
  """A data module for encapsulating the data for a Jax model to be serialized through the Orbax Model export flow."""

  def __init__(
      self,
      params: PyTree,
      apply_fn: Union[ApplyFn, Mapping[str, ApplyFn]],
      **kwargs: Any,
  ):
    pass

  @property
  def apply_fn_map(self) -> Mapping[str, ApplyFn]:
    """Returns the apply_fn_map."""
    raise NotImplementedError(
        'ObmModule.methods not implemented yet. See b/363061755.'
    )

  @property
  def model_params(self) -> PyTree:
    """Returns the model parameters."""
    raise NotImplementedError(
        'ObmModule.methods not implemented yet. See b/363061755.'
    )

  def obm_module_to_jax_exported_map(
      self,
      model_inputs: PyTree,
  ) -> Mapping[str, jax_export.Exported]:
    """Converts the OrbaxModel to jax_export.Exported."""
    raise NotImplementedError(
        'ObmModule.methods not implemented yet. See b/363061755.'
    )

  @property
  def with_gradient(self) -> bool:
    """Returns True if a gradient function is defined."""
    raise NotImplementedError(
        'ObmModule.methods not implemented yet. See b/363061755.'
    )

  @property
  def methods(self) -> Mapping[str, Callable[..., Any]]:
    """Named methods in the context of the chosen export pathway."""
    raise NotImplementedError(
        'ObmModule.methods not implemented yet. See b/363061755.'
    )
