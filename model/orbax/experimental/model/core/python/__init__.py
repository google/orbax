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

"""ML Exported Model: a lightweight library to generate TensorFlow SavedModel.

ML Exported Model is a lightweight library to generate TensorFlow (TF)
SavedModel. Its dependency on TF is minimal: only the protobuf schemas
needed by SavedModel. It aspires to evolve into a framework-agnostic
machine-learning (ML) deployment artifact that can be generated from
many ML frameworks such as TF, JAX and PyTorch.
"""

# pylint: disable=g-importing-member

from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.protos import simple_orchestration_pb2
from orbax.experimental.model.core.protos import type_pb2
from orbax.experimental.model.core.python import simple_orchestration
from orbax.experimental.model.core.python import tracing
from orbax.experimental.model.core.python import tree_util
from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.concrete_function import dtype_from_np_dtype
from orbax.experimental.model.core.python.concrete_function import dtype_to_np_dtype
from orbax.experimental.model.core.python.concrete_function import partial
from orbax.experimental.model.core.python.concrete_function import Tensor
from orbax.experimental.model.core.python.concrete_function import Variable
from orbax.experimental.model.core.python.function import Function
from orbax.experimental.model.core.python.function import np_dtype_to_shlo_dtype
from orbax.experimental.model.core.python.function import shlo_dtype_to_np_dtype
from orbax.experimental.model.core.python.function import ShloDimSize
from orbax.experimental.model.core.python.function import ShloDType
from orbax.experimental.model.core.python.function import ShloShape
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.manifest_constants import *
from orbax.experimental.model.core.python.polymorphic_function import function
from orbax.experimental.model.core.python.polymorphic_function import PolymorphicFunction
from orbax.experimental.model.core.python.save_lib import save
from orbax.experimental.model.core.python.save_lib import SaveOptions
from orbax.experimental.model.core.python.save_lib import SupplementalInfo
from orbax.experimental.model.core.python.saveable import Saveable
from orbax.experimental.model.core.python.serializable_function import SerializableFunction
from orbax.experimental.model.core.python.shlo_function import ShloFunction
from orbax.experimental.model.core.python.shlo_function import ShloFunctionSupplementalInfo
from orbax.experimental.model.core.python.signature import DType
from orbax.experimental.model.core.python.signature import OpSharding
from orbax.experimental.model.core.python.signature import Shape
from orbax.experimental.model.core.python.signature import Signature
from orbax.experimental.model.core.python.signature import TensorSpec
# TODO(wangpeng): Don't expose individual symbols from
#   simple_orchestration.py, because simple_orchestration.py will be
#   moved out of model/core/ .
from orbax.experimental.model.core.python.test_utils import ObmTestCase
from orbax.experimental.model.core.python.tree_util import Tree
from orbax.experimental.model.core.python.type_proto_util import manifest_type_to_shlo_tensor_spec_pytree
from orbax.experimental.model.core.python.type_proto_util import to_function_signature_proto
from orbax.experimental.model.core.python.unstructured_data import UnstructuredData
from orbax.experimental.model.core.python.unstructured_data import UnstructuredDataWithExtName
from orbax.experimental.model.core.python.value import ExternalValue
from orbax.experimental.model.core.python.value import Value

# pylint: enable=g-importing-member
