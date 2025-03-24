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

"""Orbax Future class used for duck typing."""

# pylint: disable=g-importing-member, unused-import

from orbax.checkpoint._src.futures.future import ChainedFuture
from orbax.checkpoint._src.futures.future import CommitFutureAwaitingContractedSignals
from orbax.checkpoint._src.futures.future import Future
from orbax.checkpoint._src.futures.future import make_async
from orbax.checkpoint._src.futures.future import NoopFuture
