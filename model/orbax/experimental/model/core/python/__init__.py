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

"""ML Exported Model: a lightweight library to generate TensorFlow SavedModel.

ML Exported Model is a lightweight library to generate TensorFlow (TF)
SavedModel. Its dependency on TF is minimal: only the protobuf schemas
needed by SavedModel. It aspires to evolve into a framework-agnostic
machine-learning (ML) deployment artifact that can be generated from
many ML frameworks such as TF, JAX and PyTorch.
"""

# pylint: disable=g-importing-member
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.protos import type_pb2
from orbax.experimental.model.core.python import tree_util
from orbax.experimental.model.core.python.constants import *
from orbax.experimental.model.core.python.function import Function
from orbax.experimental.model.core.python.function import np_dtype_to_shlo_dtype
from orbax.experimental.model.core.python.function import Sharding
from orbax.experimental.model.core.python.function import shlo_dtype_to_np_dtype
from orbax.experimental.model.core.python.function import ShloDimSize
from orbax.experimental.model.core.python.function import ShloDType
from orbax.experimental.model.core.python.function import ShloShape
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.manifest_constants import *
from orbax.experimental.model.core.python.persistence_lib import GlobalSupplemental
from orbax.experimental.model.core.python.persistence_lib import load
from orbax.experimental.model.core.python.persistence_lib import save
from orbax.experimental.model.core.python.persistence_lib import SaveOptions
from orbax.experimental.model.core.python.saveable import Saveable
from orbax.experimental.model.core.python.serializable_function import SerializableFunction
from orbax.experimental.model.core.python.shlo_function import ShloFunction
from orbax.experimental.model.core.python.shlo_function import ShloFunctionSupplementalInfo
from orbax.experimental.model.core.python.test_utils import ObmTestCase
from orbax.experimental.model.core.python.tree_util import Tree
from orbax.experimental.model.core.python.type_proto_util import from_type_proto
from orbax.experimental.model.core.python.type_proto_util import to_function_signature_proto
from orbax.experimental.model.core.python.unstructured_data import UnstructuredData
from orbax.experimental.model.core.python.unstructured_data import UnstructuredDataWithExtName
from orbax.experimental.model.core.python.value import ExternalValue
from orbax.experimental.model.core.python.value import Value
# pylint: enable=g-importing-member

OpSharding = Sharding
