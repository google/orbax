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


################################################################################
# Keyword args
################################################################################

# kwargs key for obtaining the jax2obm_kwargs.
JAX2OBM_KWARGS = 'jax2obm_kwargs'

# Jax2obm_kwargs key for the partition specs to use when converting the jax
# function to an obm function.
PSPECS = 'pspecs'

# Jax2obm_kwargs key for the native serialization platforms to use when
# converting the jax function to an obm function.
NATIVE_SERIALIZATION_PLATFORMS = 'native_serialization_platforms'

# Jax2obm_kwargs key for the checkpoint path to use when setting the checkpoint
# attribute on the OrbaxModule.
CHECKPOINT_PATH = 'checkpoint_path'

# Jax2obm_kwargs key for the weights name to use when setting the checkpoint
# attribute on the OrbaxModule.
WEIGHTS_NAME = 'weights_name'

# Jax2obm_kwargs key for input polymorphic constraints.
POLYMORPHIC_CONSTRAINTS = 'polymorphic_constraints'

# Default weights name to use if a checkpoint path is provided but a weights_
# name kwarg was not provided in the jax2obm_kwargs.
DEFAULT_WEIGHTS_NAME = 'weights'

# Jax2obm_kwargs key that triggers loading all checkpoint weights
# for the exported functions. By default, only weights used by the function
# are loaded. If this key is set to True, all weights in the checkpoint
# will be loaded. This may result in argument mismatches if the checkpoint
# contains more weights than required by the function.
LOAD_ALL_CHECKPOINT_WEIGHTS = 'load_all_checkpoint_weights'

DEFAULT_PRE_PROCESSOR_NAME = 'pre_processor'

DEFAULT_POST_PROCESSOR_NAME = 'post_processor'

DEFAULT_SUPPLEMENTAL_FILENAME = 'orchestration.pb'

# Orbax MIME types
ORBAX_CHECKPOINT_MIME_TYPE = 'application/x.orbax-checkpoint'


# XLA flags per platform for the model. This is a mapping from platform name to
# a list of xla flags. This is used to set the XLA compilation options for the
# model. If not provided, the default XLA flags for default platform (TPU) will
# be used.
XLA_FLAGS_PER_PLATFORM = 'xla_flags_per_platform'

# Mesh for the model.
JAX_MESH = 'jax_mesh'

# Whether to persist XLA flags in the model.
PERSIST_XLA_FLAGS = 'persist_xla_flags'

# Whether to enable bf16 optimization for the model.
# TODO_REGEX: b/422170690: (1): Apply this flag to the pre/post processors. (2):
# Adding filter flags once the flag is applied to the pre/post processors.
ENABLE_BF16_OPTIMIZATION = 'enable_bf16_optimization'

################################################################################
# Proto field names
################################################################################
# Proto field names for ConverterOptionsV2.BatchOptionsV2
PROTO_FIELD_BATCHOPTIONSV2_BATCH_COMPONENT = 'batch_component'
PROTO_FIELD_BATCHOPTIONSV2_SIGNATURE_NAME = 'signature_name'


################################################################################
# Logging Messages
################################################################################
# Prefix to use for prepending to logging messages
LOG_PREFIX = '[OrbaxExport]'
