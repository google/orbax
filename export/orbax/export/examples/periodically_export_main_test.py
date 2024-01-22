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

import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
from orbax.export.examples import periodically_export_main
import tensorflow as tf


_FLAGS = flags.FLAGS


class SavedModelMainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._export_base_dir = self.create_tempdir().full_path

  def test_export_periodically(self):
    with flagsaver.flagsaver(
        export_base_dir=self._export_base_dir,
        train_steps=101,
        export_interval=50,
    ):
      periodically_export_main.main([])

    x = np.array([[0.1], [0.2]])
    answer = np.sin(x)

    for step in (50, 100):
      model = tf.saved_model.load(
          os.path.join(self._export_base_dir, str(step))
      )
      prediction = model.signatures["serving_default"](x=x)["y"]
      logging.info(
          "At step: %d, prediction=%s, answer=%s", step, prediction, answer
      )


if __name__ == "__main__":
  _FLAGS.set_default("export_base_dir", "default-in-test")
  absltest.main()
