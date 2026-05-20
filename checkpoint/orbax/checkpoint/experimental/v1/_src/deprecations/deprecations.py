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

_FuncT = TypeVar('_FuncT', bound=Callable[..., Any])


def deprecated(*, new: _FuncT) -> Callable[[_FuncT], _FuncT]:
  """Decorator to mark a function as a deprecated alias of another function.

  Usage:
    @deprecated(new=new_function)
    def old_function(...): ...

  Args:
    new: The function to use instead.

  Returns:
    A decorator function.
  """

  def decorator(deprecated_func: _FuncT) -> _FuncT:
    @functools.wraps(deprecated_func)
    def wrapper(*args, **kwargs):
      deprecated_func_name = getattr(deprecated_func, '__name__', 'unknown')
      new_name = getattr(new, '__name__', 'unknown')
      warnings.warn(
          f'`{deprecated_func_name}` is deprecated, use `{new_name}` instead.',
          DeprecationWarning,
          stacklevel=2,
      )
      return deprecated_func(*args, **kwargs)

    return wrapper

  return decorator
