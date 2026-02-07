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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jd2obm import utils


@dataclasses.dataclass(frozen=True)
class MockJDSignature:
  dtype: np.dtype
  shape: tuple[int, ...]


class UtilsTest(parameterized.TestCase):

  def test_jd_signature_to_obm_spec(self):
    jd_sig = {
        'a': MockJDSignature(shape=(1, 2), dtype=np.dtype(np.int32)),
        'b': MockJDSignature(shape=(3,), dtype=np.dtype(np.float32)),
    }
    obm_spec = utils.jd_signature_to_obm_spec(jd_sig)
    expected_obm_spec = {
        'a': obm.ShloTensorSpec(shape=(1, 2), dtype=obm.ShloDType.i32),
        'b': obm.ShloTensorSpec(shape=(3,), dtype=obm.ShloDType.f32),
    }
    self.assertEqual(obm_spec, expected_obm_spec)

  def test_jd_to_obm_dtype_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "Expected a numpy.dtype, got <class 'int'> of type <class 'type'>",
    ):
      utils._jd_to_obm_dtype(int)


if __name__ == '__main__':
  absltest.main()
