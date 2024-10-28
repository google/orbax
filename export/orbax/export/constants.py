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

"""Constants used in Orbax export."""

import enum


class ExportModelType(enum.Enum):
  """Defines the export types available for serializing a JAX model through Orbax."""

  TF_SAVEDMODEL = 'TF_SAVEDMODEL'
  ORBAX_MODEL = 'ORBAX_MODEL'


# Default dictionary key used for mapping from an apply fn provided during the
# JaxModule creation if the apply function passed wasn't already provided in
# a dictionary from name to function. During creation the JaxModule will
# normalize the JaxModule arguments to be a dictionary from name to function. If
# the user provided a single function, it will be mapped to this default key.
DEFAULT_METHOD_KEY = 'jax_module_default_method'


class OrbaxNativeSerializationType(enum.Enum):
  """Defines the native serialization types available for a JAX model through Orbax."""

  CPU = ('cpu',)
  CUDA = ('cuda',)
  ROCM = ('rocm',)
  TPU = ('tpu',)


################################################################################
# Keyword args
################################################################################

# kwargs key for obtaining the jax2obm_kwargs.
JAX2OBM_KWARGS = 'jax2obm_kwargs'

# Jax2obm_kwargs key for the partition specs to use when converting the jax
# function to an obm function.
PSPECS = 'pspecs'

# Jax2obm_kwargs key for the native serialization platform to use when
# converting the jax function to an obm function.
NATIVE_SERIALIZATION_PLATFORM = 'native_serialization_platform'

# Jax2obm_kwargs key for whether to flatten the signature of the function
# during conversion.
FLATTEN_SIGNATURE = 'flatten_signature'

# Jax2obm_kwargs key for the checkpoint path to use when setting the checkpoint
# attribute on the OrbaxModule.
CHECKPOINT_PATH = 'checkpoint_path'

# Jax2obm_kwargs key for the weights name to use when setting the checkpoint
# attribute on the OrbaxModule.
WEIGHTS_NAME = 'weights_name'

# Default weights name to use if a checkpoint path is provided but a weights_
# name kwarg was not provided in the jax2obm_kwargs.
DEFAULT_WEIGHTS_NAME = 'weights'

DEFAULT_PRE_PROCESSOR_NAME = 'pre_processor'

DEFAULT_POST_PROCESSOR_NAME = 'post_processor'

DEFAULT_SUPPLEMENTAL_FILENAME = 'orchestration.pb'

# Orbax MIME types
ORBAX_CHECKPOINT_MIME_TYPE = 'orbax_checkpoint'
