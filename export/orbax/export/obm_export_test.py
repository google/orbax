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

from collections.abc import Mapping, Sequence
import contextlib
import importlib
import os
from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from orbax.export import config
from orbax.export import constants
from orbax.export import export_testing_utils
from orbax.export import jax_module
from orbax.export import obm_configs
from orbax.export import obm_export
from orbax.export import oex_orchestration
from orbax.export import serving_config
from orbax.export import utils
import tensorflow as tf


# TODO(b/363033166): Remove this function once TF isolation is done.
def _package_jax_module(m: jax_module.JaxModule):
  result = tf.Module()
  result.computation_module = m
  return result


def _assert(b):
  assert b


class ObmExportUnitTest(parameterized.TestCase, tf.test.TestCase):

  def test_incorrect_export_version(self):
    pass


if __name__ == '__main__':
  absltest.main()
