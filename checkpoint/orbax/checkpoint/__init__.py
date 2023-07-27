# Copyright 2023 The Orbax Authors.
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

import contextlib
import functools

from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import msgpack_utils
from orbax.checkpoint import test_utils
from orbax.checkpoint import transform_utils
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
from orbax.checkpoint.abstract_checkpointer import AbstractCheckpointer
from orbax.checkpoint.array_checkpoint_handler import ArrayCheckpointHandler
from orbax.checkpoint.async_checkpoint_handler import AsyncCheckpointHandler
from orbax.checkpoint.async_checkpointer import async_checkpointer_context
from orbax.checkpoint.async_checkpointer import AsyncCheckpointer
from orbax.checkpoint.checkpoint_handler import CheckpointHandler
from orbax.checkpoint.checkpoint_manager import checkpoint_manager_context
from orbax.checkpoint.checkpoint_manager import CheckpointManager
from orbax.checkpoint.checkpoint_manager import CheckpointManagerOptions
from orbax.checkpoint.checkpointer import Checkpointer
from orbax.checkpoint.checkpointer import checkpointer_context
from orbax.checkpoint.future import Future
from orbax.checkpoint.json_checkpoint_handler import JsonCheckpointHandler
from orbax.checkpoint.pytree_checkpoint_handler import ArrayRestoreArgs
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler
from orbax.checkpoint.pytree_checkpoint_handler import RestoreArgs
from orbax.checkpoint.pytree_checkpoint_handler import SaveArgs
from orbax.checkpoint.transform_utils import apply_transformations
from orbax.checkpoint.transform_utils import merge_trees
from orbax.checkpoint.transform_utils import RestoreTransform
from orbax.checkpoint.transform_utils import Transform


try:
  __IPYTHON__
  _in_ipython_session = True
except NameError:
  _in_ipython_session = False
if _in_ipython_session:
  import nest_asyncio
  nest_asyncio.apply()

# Convenient shorthand where instead of the following:
#   `checkpointer = Checkpointer(PyTreeCheckpointer())`
# we can just use:
#   `checkpointer = PyTreeCheckpointer()`
PyTreeCheckpointer = functools.partial(Checkpointer, PyTreeCheckpointHandler())

# A new PyPI release will be pushed everytime `__version__` is increased.
__version__ = '0.3.1'
