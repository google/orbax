# Copyright 2025 The Orbax Authors.
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

"""global orbax.export configuration.

There are several ways to config those global variables.
1. Initialize the ENV variable through command or at the beginning of the main
python function.

```
os.environ['OBX_EXPORT_TF_PREPROCESS_ONLY'] = 'false'
obx_export_config = obx_export.config.config
print(obx_export_config.obx_export_tf_preprocess_only)
```

2. Users can update option values using the update method. use code like

```
obx_export_config = obx_export.config.config
obx_export_config.update('obx_export_tf_preprocess_only', True)
```
3.A context manager allows for thread-local overrides of global settings.
Option values can be initially set via environment variables. code like

```
with obx_export.config.obx_export_tf_preprocess_only(True):
  ...
```
"""

import contextlib
import os
import threading
from typing import Any, Callable, Optional


class _Unset:
  pass


# A sentinel object to indicate that a config option is unset.
unset = _Unset()

_thread_local_state = threading.local()


def bool_env(varname: str, default: bool) -> bool:
  """Read an environment variable and interpret it as a boolean.

  True values are (case insensitive): 'y', 'yes', 't', 'true', 'on', and '1';
  false values are 'n', 'no', 'f', 'false', 'off', and '0'.

  Args:
    varname: the name of the variable
    default: the default boolean value

  Returns:
    True or False

  Raises: ValueError if the environment variable is anything else.
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError(f'invalid truth value {val!r} for environment {varname!r}')


# TODO: b/335476191 - support Config class update value through absl flags.


class Config:
  """Class for managing configuration options."""

  def __init__(self):
    self.values: dict[str, Any] = {}
    self.meta: dict[str, Any] = {}
    self._update_hooks: dict[str, Callable[[Any], None]] = {}

  def _undefine_state(self, name: str) -> None:
    """Unset a config option."""
    # This method is for unittest only most time.
    if name not in self.values:
      raise AttributeError(f'Unrecognized config option: {name}')
    del self.values[name]
    if name in self.meta:
      del self.meta[name]
    if name in self._update_hooks:
      del self._update_hooks[name]

    # Deregistering is on the `Config` class
    delattr(Config, name)

  def update(self, name: str, val: Any) -> None:
    """Update the value of a config option."""
    if name not in self.values:
      raise AttributeError(f'Unrecognized config option: {name}')
    self.values[name] = val

    hook = self._update_hooks.get(name, None)
    if hook:
      hook(val)

  def read(self, name):
    """Read the value of a config option."""
    try:
      return self.values[name]
    except KeyError as e:
      raise AttributeError(f'Unrecognized config option: {name}') from e

  def _add_option(
      self,
      name,
      default,
      opt_type,
      meta_args,
      meta_kwargs,
      update_hook: Optional[Callable[[Any], None]] = None,
  ):
    """Add a config option.

    Args:
      name: the name of the config option
      default: the default value
      opt_type: the type of the config option
      meta_args: the args to pass to the meta function
      meta_kwargs: the kwargs to pass to the meta function
      update_hook: a function to call when the config option is updated
    """
    if name in self.values:
      raise ValueError(f'Config option {name} already defined')
    self.values[name] = default
    self.meta[name] = (opt_type, meta_args, meta_kwargs)
    if update_hook:
      self._update_hooks[name] = update_hook
      update_hook(default)

  def define_bool(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop('update_hook', None)
    self._add_option(name, default, bool, args, kwargs, update_hook=update_hook)

  def define_bool_state(
      self,
      name: str,
      default: bool,
      help_str: str,
      *,
      update_global_hook: Optional[Callable[[bool], None]] = None,
      update_thread_local_hook: Optional[
          Callable[[Optional[bool]], None]
      ] = None,
  ) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    name = name.lower()
    self.define_bool(
        name,
        bool_env(name.upper(), default),
        help_str,
        update_hook=update_global_hook,
    )

    def get_state(self):
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self.read(name)

    # New option registering is on the `Config` class.
    setattr(Config, name, property(get_state))

    return _StateContextManager(
        name,
        help_str,
        update_thread_local_hook,
        default_value=True,
    )


class _StateContextManager:
  """Context manager for managing thread-local state."""

  def __init__(
      self,
      name,
      help_str,
      update_thread_local_hook,
      default_value: Any,
  ):
    self._name = name
    self.__name__ = name
    self.__doc__ = f'Context manager for `{name}` config option.\n\n{help_str}'
    self._update_thread_local_hook = update_thread_local_hook
    self._default_value = default_value

  def _maybe_update_thread_local_hook(self, new_val: Any):
    if self._update_thread_local_hook:
      self._update_thread_local_hook(new_val)

  @contextlib.contextmanager
  def __call__(self, new_val: Any):
    prev_val = getattr(_thread_local_state, self._name, unset)
    setattr(_thread_local_state, self._name, new_val)
    self._maybe_update_thread_local_hook(new_val)
    try:
      yield
    finally:
      if prev_val is unset:
        delattr(_thread_local_state, self._name)
        if self._update_thread_local_hook:
          self._update_thread_local_hook(None)
      else:
        setattr(_thread_local_state, self._name, prev_val)
        if self._update_thread_local_hook:
          self._update_thread_local_hook(prev_val)


config = Config()

obx_export_tf_preprocess_only = config.define_bool_state(
    name='obx_export_tf_preprocess_only',
    default=bool_env('OBX_EXPORT_TF_PREPROCESS_ONLY', False),
    help_str=(
        'If it is True, the export will only export the '
        'servering_config.tf_preprocess instead of the whole model. This mode '
        'is majorly used for debugging.'
    ),
)
