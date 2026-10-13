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

"""Defines exported :py:class:`CheckpointArgs` classes.

:py:class:`.CheckpointHandler` subclasses define logic used to save and restore
an object to and from a checkpoint. Each :py:class:`.CheckpointHandler`
has corresponding :py:class:`.SaveArgs` and :py:class:`.RestoreArgs`
classes that define the arguments used to call the handler.

The `ocp.args` module provides a complete definition of these classes. Refer to
`ocp.handlers` for more information on the handlers themselves.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0 import args as _mod
from orbax.checkpoint.v0.args import *

sys.modules[__name__] = _mod
