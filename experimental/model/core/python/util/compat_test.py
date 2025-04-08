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

# Copyright 2024 The Orbax Model Authors. All Rights Reserved.
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
# ==============================================================================
"""Compat tests."""

from orbax.experimental.model.core.python.util import compat

from absl.testing import absltest


class CompatTest(googletest.TestCase):

  def testCompatValidEncoding(self):
    self.assertEqual(compat.as_bytes("hello", "utf8"), b"hello")
    self.assertEqual(compat.as_text(b"hello", "utf-8"), "hello")

  def testCompatInvalidEncoding(self):
    with self.assertRaises(LookupError):
      compat.as_bytes("hello", "invalid")

    with self.assertRaises(LookupError):
      compat.as_text(b"hello", "invalid")


if __name__ == "__main__":
  googletest.main()
