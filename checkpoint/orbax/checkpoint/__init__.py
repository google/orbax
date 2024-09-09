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

"""Defines exported symbols for the namespace package `orbax.checkpoint`."""

import asyncio
import contextlib
import functools
import nest_asyncio

from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import args
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import logging
from orbax.checkpoint import metadata
from orbax.checkpoint import msgpack_utils
from orbax.checkpoint import multihost
from orbax.checkpoint import options
from orbax.checkpoint import path
from orbax.checkpoint import serialization
from orbax.checkpoint import test_utils
from orbax.checkpoint import transform_utils
from orbax.checkpoint import tree
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
from orbax.checkpoint.path import step

# pylint: disable=g-importing-member, g-bad-import-order
from orbax.checkpoint.abstract_checkpoint_manager import AbstractCheckpointManager
from orbax.checkpoint.abstract_checkpointer import AbstractCheckpointer

from orbax.checkpoint.async_checkpointer import AsyncCheckpointer
from orbax.checkpoint.checkpoint_manager import AsyncOptions
from orbax.checkpoint.checkpoint_manager import CheckpointManager
from orbax.checkpoint.checkpoint_manager import CheckpointManagerOptions
from orbax.checkpoint.checkpointer import Checkpointer
from orbax.checkpoint.future import Future
from orbax.checkpoint.pytree_checkpointer import PyTreeCheckpointer
from orbax.checkpoint.standard_checkpointer import StandardCheckpointer
from orbax.checkpoint.transform_utils import apply_transformations
from orbax.checkpoint.transform_utils import merge_trees
from orbax.checkpoint.transform_utils import RestoreTransform
from orbax.checkpoint.transform_utils import Transform
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import RestoreArgs
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import SaveArgs

from orbax.checkpoint.handlers import *
# pylint: enable=g-importing-member, g-bad-import-order

try:
  asyncio.get_running_loop()
  nest_asyncio.apply()
except RuntimeError:
  pass

# pylint: disable=g-bad-import-order, g-import-not-at-top
from orbax.checkpoint import version
# A new PyPI release will be pushed everytime `__version__` is increased.
__version__ = version.__version__
del version
# pylint: enable=g-bad-import-order, g-import-not-at-top
