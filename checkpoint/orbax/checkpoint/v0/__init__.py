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

"""Defines exported symbols for the namespace package `orbax.checkpoint.v0`."""

# pylint: disable=g-importing-member, g-bad-import-order, g-import-not-at-top, unused-import

import contextlib
import functools

from orbax.checkpoint.v0 import arrays
from orbax.checkpoint.v0 import aggregate_handlers
from orbax.checkpoint.v0 import args
from orbax.checkpoint.v0 import checkpoint_utils
from orbax.checkpoint.v0 import checkpointers
from orbax.checkpoint.v0 import checkpoint_managers
from orbax.checkpoint.v0 import handlers
from orbax.checkpoint.v0 import logging
from orbax.checkpoint.v0 import metadata
from orbax.checkpoint.v0 import msgpack_utils
from orbax.checkpoint.v0 import options
from orbax.checkpoint.v0 import path
from orbax.checkpoint.v0 import pathways
from orbax.checkpoint.v0 import serialization
from orbax.checkpoint.v0 import transform_utils
from orbax.checkpoint.v0 import tree
from orbax.checkpoint.v0 import type_handlers
from orbax.checkpoint.v0 import utils
from orbax.checkpoint.v0 import version

# TODO(cpgaffney): Import the public multihost API.
from orbax.checkpoint.v0._src.multihost import multihost
from orbax.checkpoint.v0._src.path import step

from orbax.checkpoint.v0.future import Future

from orbax.checkpoint.v0.transform_utils import apply_transformations
from orbax.checkpoint.v0.transform_utils import merge_trees
from orbax.checkpoint.v0.transform_utils import RestoreTransform
from orbax.checkpoint.v0.transform_utils import Transform

from orbax.checkpoint.v0.abstract_checkpoint_manager import AbstractCheckpointManager
from orbax.checkpoint.v0.checkpointers import *
from orbax.checkpoint.v0.checkpoint_manager import CheckpointManager
from orbax.checkpoint.v0.checkpoint_manager import AsyncOptions
from orbax.checkpoint.v0.checkpoint_manager import CheckpointManagerOptions

from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import RestoreArgs
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import ArrayRestoreArgs
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import SaveArgs

# Important handlers.
from orbax.checkpoint.v0._src.handlers.checkpoint_handler import CheckpointHandler
from orbax.checkpoint.v0._src.handlers.async_checkpoint_handler import AsyncCheckpointHandler

from orbax.checkpoint.v0._src.handlers.array_checkpoint_handler import ArrayCheckpointHandler
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import CompositeCheckpointHandler
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import CompositeOptions
from orbax.checkpoint.v0._src.handlers.json_checkpoint_handler import JsonCheckpointHandler
from orbax.checkpoint.v0._src.handlers.proto_checkpoint_handler import ProtoCheckpointHandler
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import PyTreeCheckpointHandler
from orbax.checkpoint.v0._src.handlers.random_key_checkpoint_handler import JaxRandomKeyCheckpointHandler
from orbax.checkpoint.v0._src.handlers.random_key_checkpoint_handler import NumpyRandomKeyCheckpointHandler
from orbax.checkpoint.v0._src.handlers.standard_checkpoint_handler import StandardCheckpointHandler

from orbax.checkpoint.v0._src.handlers.handler_registration import CheckpointHandlerRegistry
from orbax.checkpoint.v0._src.handlers.handler_registration import DefaultCheckpointHandlerRegistry

# This class should be regarded as internal-only, and may be removed without
# warning.
from orbax.checkpoint.v0._src.handlers.base_pytree_checkpoint_handler import BasePyTreeCheckpointHandler

# Test utils.
from orbax.checkpoint.v0 import test_utils

from orbax.checkpoint.v0._src.serialization.type_handlers import PLACEHOLDER

# A new PyPI release will be pushed everytime `__version__` is increased.
__version__ = version.__version__
del version

from orbax.checkpoint.v0._src.logging import monitoring as _orbax_monitoring

_orbax_monitoring.initialize_from_env()
del _orbax_monitoring
