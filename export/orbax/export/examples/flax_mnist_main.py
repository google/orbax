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

r"""Export a MNIST JAX model.

python flax_mnist_main.py --output_dir=<OUTPUT_DIR>
"""
from absl import app
from absl import flags
from absl import logging

import flax.linen as nn
import jax
import jax.numpy as jnp
from orbax.export import ExportManager
from orbax.export import JaxModule
from orbax.export import ServingConfig
import tensorflow as tf

_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Diretory to export the model.', required=True)


class JaxMnist(nn.Module):
  """Mnist model."""

  @nn.compact
  def __call__(self, x):
    """See base class."""
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x


def export_mnist() -> None:
  """Exports a Mnist JAX model."""
  batch_size = _BATCH_SIZE.value
  model_path = _OUTPUT_DIR.value

  # Initialize the model.
  model = JaxMnist()
  params = model.init(jax.random.PRNGKey(123), jnp.ones((1, 28, 28, 1)))

  # Wrap the model params and function into a JaxModule.
  jax_module = JaxModule(
      params,
      model.apply,
      trainable=False,
      input_polymorphic_shape='(b, ...)' if batch_size is None else None)

  # Specify the serving configuration and export the model.
  em = ExportManager(jax_module, [
      ServingConfig(
          'serving_default',
          input_signature=[
              tf.TensorSpec([batch_size, 28, 28, 1], tf.float32, name='inputs')
          ],
          tf_postprocessor=lambda x: dict(outputs=x)),
  ])
  # Save the model.
  logging.info('Exporting the model to %s.', model_path)
  em.save(model_path)

  # Test that the saved model could be loaded and run.
  logging.info('Loading the model from %s.', model_path)
  loaded = tf.saved_model.load(model_path)
  logging.info('Loaded the model from %s.', model_path)

  inputs = jnp.ones((batch_size or 1, 28, 28, 1))
  savedmodel_output = loaded.signatures['serving_default'](inputs=inputs)
  jax_output = model.apply(params, inputs)

  logging.info('Savemodel output: %s, JAX output: %s', savedmodel_output,
               jax_output)


if __name__ == '__main__':
  app.run(lambda _: export_mnist())
