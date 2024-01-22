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

"""Example of exporting a toy JAX model periodically during training."""

from collections.abc import Sequence
import functools
import os

from absl import app
from absl import flags
from absl import logging
from flax import linen as nn
import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
import optax
from orbax import export as oex
import tensorflow as tf


ArrayTree = jaxtyping.PyTree[jax.typing.ArrayLike]


_EXPORT_BASE_DIR = flags.DEFINE_string(
    "export_base_dir",
    None,
    "Base directory for exporting models.",
    required=True,
)
_TRAIN_STEPS = flags.DEFINE_integer(
    "train_steps",
    100,
    "Number of training steps.",
)
_EXPORT_INTERVAL = flags.DEFINE_integer(
    "export_interval",
    20,
    "Export model every export_interval steps.",
)


class MLP(nn.Module):
  """A multi-layer perceptron."""

  feature_sizes: Sequence[int]
  dropout_rate: float = 0

  @nn.compact
  def __call__(self, inputs: jax.typing.ArrayLike, training: bool):
    """See base class."""
    x = inputs
    for size in self.feature_sizes[:-1]:
      x = nn.Dense(features=size)(x)
      x = nn.relu(x)
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
    x = nn.Dense(features=self.feature_sizes[-1])(x)
    return x


def mse(
    params: ArrayTree,
    rng_key: jax.Array,
    x_batched: jax.typing.ArrayLike,
    y_batched: jax.typing.ArrayLike,
    model: nn.Module,
) -> jax.typing.ArrayLike:
  """Computes the mean squared error for a batch of training data."""
  predictions = model.apply(
      params, x_batched, training=True, rngs={"dropout": rng_key}
  )
  return jnp.mean(optax.losses.squared_error(predictions, y_batched)) * 0.5


def generate_samples(
    size: int,
) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
  """Generates a batch of samples."""
  x = np.random.randn(size, 1)
  y = np.sin(x) + np.random.randn(size, 1) * 0.001
  return x, y


def train(
    model: nn.Module,
    x_samples: jax.typing.ArrayLike,
    y_samples: jax.typing.ArrayLike,
    steps: int,
    export_base_dir: str,
    export_interval: int = 20,
):
  """Trains a model on a batch of samples."""
  init_key, key = jax.random.split(jax.random.key(0))
  params = model.init(init_key, x_samples, training=False)
  optimizer = optax.adam(learning_rate=1e-2)
  opt_state = optimizer.init(params)

  @jax.jit
  def train_step(
      params: ArrayTree,
      opt_state: ArrayTree,
      key: jax.Array,
      x_samples: jax.typing.ArrayLike,
      y_samples: jax.typing.ArrayLike,
  ):
    dropout_key, key = jax.random.split(key)
    loss_val, grads = jax.value_and_grad(functools.partial(mse, model=model))(
        params, dropout_key, x_samples, y_samples
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val, key

  jax_module = oex.JaxModule(
      params,
      functools.partial(model.apply, training=False),
      input_polymorphic_shape="b, ...",
  )
  serving_config = oex.ServingConfig(
      "serving_default",
      input_signature=[
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name="x")
      ],
      tf_postprocessor=lambda out: {"y": out},
  )
  export_mgr = oex.ExportManager(jax_module, [serving_config])

  for i in range(steps):
    params, opt_state, loss_val, key = train_step(
        params, opt_state, key, x_samples, y_samples
    )

    if i % 20 == 0:
      logging.info("Loss step %d: loss=%f.", i, loss_val)

    if i > 0 and i % export_interval == 0:
      jax_module.update_variables(params)
      export_path = os.path.join(export_base_dir, str(i))
      export_mgr.save(export_path)
      logging.info("Model exported to %s", export_path)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  model = MLP(feature_sizes=[32, 32, 1], dropout_rate=0.1)
  x_samples, y_samples = generate_samples(size=128)
  train(
      model,
      x_samples,
      y_samples,
      _TRAIN_STEPS.value,
      _EXPORT_BASE_DIR.value,
      _EXPORT_INTERVAL.value,
  )


if __name__ == "__main__":
  app.run(main)
