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

"""Utils for creating and finalizing temporary paths.

Note that the configurability provided by this feature does
not leave users free to define their own temporary path structure. The current
implementation is mainly a refactoring of old logic that separately created
temp directories and finalized them. It does not touch other logic that detects
temp checkpoints and cleans them up (primarily located in
orbax.checkpoint.path.step and :py:class:`.CheckpointManager`).

Ordinarily, atomic logic defaults to :py:class:`AtomicRenameTemporaryPath`,
which uses an atomic rename to indicate checkpoint completion. However, not all
filesystems support atomic rename, so :py:class:`CommitFileTemporaryPath` is
provided as an alternative, which uses a "commit_success" file to indicate
completion.

Ideally, we would standardize on a single behavior, but it is difficult, largely
for legacy reasons, to achieve this. Furthermore, there are many other
alternative ways of ensuring save atomicity. As such, we have opted to provide
a more flexible approach that allows users to configure the behavior they want.

Configuration can be done in the following way::

  AsyncCheckpointer(
      StandardCheckpointHandler(),
      temporary_path_class=CommitFileTemporaryPath,
  )

  # OR

  CheckpointManager(
      directory,
      item_names=('state', 'dataset',),
      options=CheckpointManagerOptions(
          temporary_path_class=atomicity.CommitFileTemporaryPath
      ),
  )
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.path import atomicity as _mod
from orbax.checkpoint.v0._src.path.atomicity import *
from orbax.checkpoint.v0._src.path.atomicity import _LAST_CHECKPOINT_WRITE_TIME
from orbax.checkpoint.v0._src.path.atomicity import _T
from orbax.checkpoint.v0._src.path.atomicity import _create_tmp_directory
from orbax.checkpoint.v0._src.path.atomicity import _get_tmp_directory
from orbax.checkpoint.v0._src.path.atomicity import _get_final_directory
from orbax.checkpoint.v0._src.path.atomicity import _shared_validate
from orbax.checkpoint.v0._src.path.atomicity import _create_paths

sys.modules[__name__] = _mod
