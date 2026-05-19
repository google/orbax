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

"""Defines deprecated legacy aliases for Orbax V1 API."""

import functools
from typing import Any, Callable, TypeVar
import warnings

from orbax.checkpoint.experimental.v1._src.loading import loading
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading
from orbax.checkpoint.experimental.v1._src.saving import saving

_FuncT = TypeVar('_FuncT', bound=Callable[..., Any])


def deprecated(*, new: _FuncT) -> Callable[[_FuncT], _FuncT]:
  """Decorator to mark a function as a deprecated alias of another function."""

  def decorator(deprecated_func: _FuncT) -> _FuncT:
    @functools.wraps(deprecated_func)
    def wrapper(*args, **kwargs):
      alias_name = getattr(deprecated_func, '__name__', 'unknown')
      new_name = getattr(new, '__name__', 'unknown')
      warnings.warn(
          f'`{alias_name}` is deprecated, use `{new_name}` instead.',
          DeprecationWarning,
          stacklevel=2,
      )
      return deprecated_func(*args, **kwargs)

    return wrapper

  return decorator


@deprecated(new=saving.save)
def save_pytree(*args, **kwargs):
  return saving.save(*args, **kwargs)


@deprecated(new=saving.save_async)
def save_pytree_async(*args, **kwargs):
  return saving.save_async(*args, **kwargs)


@deprecated(new=loading.load)
def load_pytree(*args, **kwargs):
  return loading.load(*args, **kwargs)


@deprecated(new=loading.load_async)
def load_pytree_async(*args, **kwargs):
  return loading.load_async(*args, **kwargs)


@deprecated(new=metadata_loading.metadata)
def pytree_metadata(*args, **kwargs):
  return metadata_loading.metadata(*args, **kwargs)
