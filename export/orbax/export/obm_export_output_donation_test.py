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

"""Tests for exporting JAX models whose outputs are written to donated buffers."""

import os
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
from orbax.export import constants
from orbax.export import jax_module
from orbax.export import obm_configs
from orbax.export import obm_export
from orbax.export import serving_config
from orbax.export import typing as orbax_export_typing

_SIGNATURE_KEY = 'my_model'
_BATCH_SIZE = 2
_IN_FEATURES = 3
_OUT_FEATURES = 4
_OUTPUT_NAMES = ('logits', 'probs')


def _params() -> dict[str, jax.Array]:
  return {
      'w': jnp.ones((_IN_FEATURES, _OUT_FEATURES), jnp.float32),
      'b': jnp.ones((_OUT_FEATURES,), jnp.float32),
  }


def _two_named_outputs(
    params: Any, inputs: dict[str, jax.Array]
) -> dict[str, jax.Array]:
  logits = inputs['x'] @ params['w'] + params['b']
  return {'logits': logits, 'probs': jax.nn.softmax(logits)}


def _one_unnamed_output(params: Any, inputs: dict[str, jax.Array]) -> jax.Array:
  return inputs['x'] @ params['w'] + params['b']


def _model(apply_fn=_two_named_outputs) -> orbax_export_typing.ApplyFnInfo:
  return orbax_export_typing.ApplyFnInfo(
      apply_fn=apply_fn,
      input_keys={'x'},
      output_keys=set(_OUTPUT_NAMES),
  )


class ObmExportOutputDonationTest(absltest.TestCase):

  # Dummy test to make copybara happy, will be removed once all the obm
  # dependencies are OSSed.
  def test_dummy(self):
    assert True


if __name__ == '__main__':
  absltest.main()
