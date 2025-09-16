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

"""Export class that implements the save and load abstract class defined in Export Base for use with the Orbax Model export format."""

from collections.abc import Callable, Mapping, Sequence
import functools
import itertools
import os
from typing import Any, Dict, Tuple, cast

from absl import logging
import jax
from jax import export as jax_export
from orbax.export import constants
from orbax.export import export_base
from orbax.export import jax_module
from orbax.export import obm_configs
from orbax.export import oex_orchestration
from orbax.export import serving_config as osc
from orbax.export import typing
from orbax.export import utils
from orbax.export.modules import obm_module
import tensorflow as tf


PyTree = typing.PyTree


class ObmExport(export_base.ExportBase):
  """Defines the save and load methods for exporting a model using Orbax Model export."""

  def __init__(
      self,
      module: jax_module.JaxModule,
      serving_configs: Sequence[osc.ServingConfig],
  ):
    """Initializes the ObmExport class."""
    if module.export_version != constants.ExportModelType.ORBAX_MODEL:
      raise ValueError(
          "JaxModule export version is not of type ORBAX_MODEL. Please use the"
          " correct export_version. Expected ORBAX_MODEL, got"
          f" {module.export_version}"
      )

    obm_model_module = module.export_module()

  def save(
      self,
      model_path: str,
      **kwargs: Any,
  ):
    """Saves a Jax model in the Orbax Model export format.

    Args:
      model_path: The path to save the model.
      **kwargs: Additional arguments to pass to the `save` method. Accepted
        arguments are `save_options` and `serving_signatures`.
    """

  def load(self, model_path: str, **kwargs: Any):
    """Loads the model previously saved in the Orbax Model export format."""
    logging.info("Loading model using Orbax Export Model.")
    raise NotImplementedError("ObmExport.load not implemented yet.")

  @property
  def serving_signatures(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map of signature keys to serving functions."""
    raise NotImplementedError("ObmExport.load not implemented yet.")
