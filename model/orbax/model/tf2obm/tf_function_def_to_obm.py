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

"""A converter from TF function to Orbax Model data structure."""

from orbax.experimental.model import core as obm
from orbax.experimental.model.tf2obm.utils import get_input_signature
from orbax.experimental.model.tf2obm.utils import get_output_signature
from orbax.experimental.model.tf2obm.utils import tf_signature_to_obm_spec
import tensorflow as tf


TF_FUNCTION_DEF_MIME_TYPE = "tf_function_def"
TF_FUNCTION_DEF_VERSION = "0.0.1"


def tf_concrete_function_to_obm_function(
    fn: tf.types.experimental.ConcreteFunction,
) -> obm.SerializableFunction:
  """Converts a TF `ConcreteFunction` to an `obm.SerializableFunction`."""
  unstructured_data = obm.manifest_pb2.UnstructuredData(
      inlined_bytes=fn.function_def.SerializeToString(),
      mime_type=TF_FUNCTION_DEF_MIME_TYPE,
      version=TF_FUNCTION_DEF_VERSION,
  )
  return obm.SerializableFunction(
      body=obm.UnstructuredDataWithExtName(
          proto=unstructured_data, ext_name="pb"
      ),
      input_signature=tf_signature_to_obm_spec(get_input_signature(fn)),
      output_signature=tf_signature_to_obm_spec(get_output_signature(fn)),
  )
