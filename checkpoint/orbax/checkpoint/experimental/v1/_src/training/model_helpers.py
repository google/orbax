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

"""Additional functions for training.py, which trains a simple CNN model with MNIST data and demonstrates how to use orbax checkpointing."""

from flax import nnx
import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

PartitionSpec = jax.sharding.PartitionSpec


class DotReluDot(nnx.Module):
  """A simple model with two linear layers and a ReLU activation."""

  def __init__(self, depth: int, rngs: nnx.Rngs):
    init_fn = nnx.initializers.lecun_normal()

    self.dot1 = nnx.Linear(
        in_features=784,
        out_features=depth,
        kernel_init=nnx.with_partitioning(init_fn, (None, 'model')),
        use_bias=False,
        rngs=rngs,
    )

    self.w2 = nnx.Param(
        init_fn(rngs.params(), (depth, depth)),
        sharding=('model', None),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x.reshape((x.shape[0], -1))
    y = self.dot1(x)
    y = jax.nn.relu(y)
    y = jax.lax.with_sharding_constraint(
        y,
        PartitionSpec(
            'data',
        ),
    )
    z = jnp.dot(y, self.w2.value)
    return z


def process_sample(sample: dict[str, jax.Array]) -> dict[str, jax.Array]:
  """Converts the image to float32 and normalizes it."""
  return {
      'image': sample['image'].astype(np.float32) / 255.0,
      'label': sample['label'],
  }


def create_dataset(split: str, batch_size: int) -> pygrain.IterDataset:
  """Creates a Grain-based dataset for a given split."""
  """TODO(zachmeyers): parameterize seed."""  # pylint: disable=pointless-string-statement
  dataset = (
      pygrain.MapDataset.source(tfds.data_source('mnist', split=split))
      .map(process_sample)
      .seed(seed=45)
      .shuffle()
      .batch(batch_size, drop_remainder=True)
  )
  return dataset.to_iter_dataset()
