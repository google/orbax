# Orbax v1 Checkpoint Configuration & Migration Guide

This guide provides a thorough overview of the Orbax v1 configuration model and
a detailed mapping from v0 `CheckpointManagerOptions`.

---

## 1. Before & After: Migration Example

Migrating from `CheckpointManager` (v0) to `Checkpointer` (v1) means moving
away from the `CheckpointManagerOptions` dataclass.

### Before: CheckpointManager (v0)

```python
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
    AsyncOptions
)

options = CheckpointManagerOptions(
    save_interval_steps=10,
    max_to_keep=5,
    step_prefix='test_run',
    async_options=AsyncOptions(timeout_secs=60),
    lightweight_initialize=True,
    cleanup_tmp_directories=True,
)

manager = CheckpointManager(
    directory,
    {'params': PyTreeCheckpointer(), 'state': PyTreeCheckpointer()},
    options=options
)

# Save & Restore
manager.save(step, {'params': params, 'state': state})
restored = manager.restore(step)
params, state = restored['params'], restored['state']
```

### After: Checkpointer (v1)

```python
import orbax.checkpoint.v1 as ocp

# 1. Environment/IO settings go in Context.
context = ocp.Context(
    async_options=ocp.options.AsyncOptions(timeout_secs=60),
)

# 2. Logic & Lifecycle settings passed to Checkpointer constructor.
with context:
  with ocp.training.Checkpointer(
      directory,
      save_decision_policy=ocp.training.save_decision_policies.FixedIntervalPolicy(
          interval=10
      ),
      preservation_policy=ocp.training.preservation_policies.LatestN(n=5),
      step_name_format=ocp.path.step.standard_name_format(step_prefix='test_run'),
      lightweight_initialize=True,
      cleanup_tmp_directories=True,
  ) as ckptr:
    # Save
    ckptr.save_checkpointables(step, {'params': params, 'state': state})

    # Restore (using 'abstract_checkpointables' to define structure/sharding)
    restored = ckptr.load_checkpointables(
        step,
        abstract_checkpointables={'params': params, 'state': state}
    )
    params, state = restored['params'], restored['state']
```

---

## 2. Orbax v1 Configuration Overview

Orbax v1 moves away from the monolithic `CheckpointManagerOptions` in favor of a
modular approach where configuration is split between the `Checkpointer`
constructor and a global `Context`.

### Why the Change?

The v1 model is designed for **separation of concerns** and **explicitness**:

- **Separation of Logic from Environment**: High-level training logic (when to
  save, what to keep) is handled by the `Checkpointer` constructor, while
  environmental/IO settings (multiprocessing, async timeouts) are isolated in
  the `Context`.
- **Modularity**: Policy and Option objects make it easier to test and reuse
  specific behaviors independently.

### 2.1 The Checkpointer Constructor

The `Checkpointer` is the primary object used in your training loop. It accepts
high-level behavior policies and basic training session settings:

- **save_decision_policy**: Determines when a checkpoint should be saved (e.g.,
  at fixed intervals, during preemption). Use policies from
  `ocp.training.save_decision_policies`.
- **preservation_policy**: Determines which historical checkpoints to keep
  (e.g.,
  latest N, every N steps). Use policies from
  `ocp.training.preservation_policies`.
- **step_name_format**: Defines the naming convention for step directories
  (e.g.,
  simple integer, prefixed). Use Name Format from
  `ocp.path.step`.
- **custom_metadata**: A JSON dictionary for root-level metadata relevant to
  the entire sequence.

### 2.2 The Orbax Context

Environment-specific and detailed I/O settings are managed via
`ocp.Context`. This allows you to set configuration once for a
training task without passing it through every call.

The `Context` is composed of several specialized options classes:

- **FileOptions**: Configures parameters for creating and managing checkpoint
  directories and files on disk (e.g., permissions, atomicity protocols, and
  custom path implementations).
- **DeletionOptions**: Configures checkpoint deletion behavior.
- **AsyncOptions**: Configures asynchronous checkpoint saving operations,
  including timeouts and post-finalization callbacks.
- **MultiprocessingOptions**: Configures multiprocessing behavior, such as
  designating the primary host and identifying active process subsets.
- **Component-Specific Options**: Provides granular control for specialized
  handlers like `PyTreeOptions` and `ArrayOptions` (e.g., storage formats,
  dtypes, and concurrent I/O limits).
- **PathwaysOptions**: Configures Pathways-specific saving and loading
  implementations.

---

## 3. v0 to v1 Mapping Guide

### Shift in Philosophy: Configuration vs. Composition
The most significant change in v1 is the move from **flat configuration
flags** to **structured policy composition**.

*   **v0 (Configuration Flags)**: You passed many independent, flat arguments
    (like `save_interval_steps`, `max_to_keep`, `should_save_fn`) to the
    `CheckpointManager`. The manager used internal "builder" logic to decide
    which flags took priority and how to combine them. This often led to
    behavior where one flag would silently disable another.
*   **v1 (Composition)**: You construct the exact logic tree you want using
    **Policy Objects** and pass those to the `Checkpointer`. The checkpointer
    has no hidden priority logic; it executes exactly what you compose. If you
    want multiple rules (e.g., "save every 10 steps" AND "save on
    preemption"), you **must** explicitly combine them using an `AnyPolicy`
    wrapper.

This shift ensures that your checkpointing logic is explicit and readable.

### 3.1 Explicitly Managed in v1 (Checkpointer Args)
These options are passed directly to the `Checkpointer` constructor.

| v0 Option | v1 Equivalent | Description |
| :--- | :--- | :--- |
| `save_decision_policy` | `save_decision_policy` | An object used to <br> determine when a checkpoint should be saved. |
| `preservation_policy` | `preservation_policy` | An object used to <br> determine which checkpoints to preserve. |
| `step_name_format` | `step_name_format` | NameFormat to build or <br> find steps under input root directory. |
| `custom_metadata` | `custom_metadata` | A JSON dictionary representing user-specified <br> custom metadata for the entire sequence. |

---

### 3.2 Mapped to SaveDecisionPolicy
This section maps v0 options related to save decisions to their v1 equivalents.
In v0, these options were used when a `save_decision_policy` was not directly
provided. In v1, these concepts are now encapsulated within specific policies
from `ocp.training.save_decision_policies`. These policies are passed directly
via the `save_decision_policy` option during `Checkpointer` initialization.
Multiple save conditions, which in v0 were configured through various separate
options, can now be combined into a single `save_decision_policy` using
`ocp.training.save_decision_policies.AnySavePolicy`.

| v0 Option | v1 Migration Logic | Description |
| :--- | :--- | :--- |
| `save_interval_steps` | `FixedIntervalPolicy(interval=int)` | The frequency at <br> which checkpoints should be saved <br> (default=1). Evaluates `step.step % interval == 0`. |
| `should_save_fn` | Implement `SaveDecisionPolicy` protocol. | Callable to check <br> if a given step can be saved. In v1, <br> this must be implemented as a class <br> obeying the protocol. |
| `save_on_steps` | `SpecificStepsPolicy(steps=Container[int])` | Optional set of specific steps at which <br> checkpoints should be saved.|
| `Multiple Save Conditions` | `AnySavePolicy([policy1, policy2])` | Evaluates a sequence of policies and saves <br> if any child policy returns `True`. |

**Notes:**

**Removal of Implicit Priority & Logic Builder**

In v0, Orbax used internal "builder" logic to prioritize certain flags. For
example, if `should_save_fn` was present, `save_interval_steps` and
`save_on_steps` were **silently ignored**.

In v1, this implicit prioritizing is **removed**. The `Checkpointer` only
executes the exact `save_decision_policy` you provide. If you want multiple
rules to apply, you **must** explicitly combine them using `AnySavePolicy`.
There are no hidden overrides or priority tiers in v1.

**v1 Migration**:

In v1, you must provide a `SaveDecisionPolicy` directly to the `Checkpointer`
constructor for custom behavior. Legacy v0 options like `save_interval_steps` or
`should_save_fn` are no longer supported as constructor arguments and must be
translated into a policy as shown above.

**Default Behavior**:

If no `save_decision_policy` is provided, Orbax v1 defaults to an
`AnySavePolicy` that combines:

1. `InitialSavePolicy()`: Saves a checkpoint at step 0 if none exist.
2. `ContinuousCheckpointingPolicy()`: Saves as frequently as possible
   (useful for async).
3. `PreemptionCheckpointingPolicy()`: Saves immediately when a preemption
   signal is detected.

#### Custom SaveDecisionPolicy Example
If you have a custom saving rule, you can implement the `SaveDecisionPolicy`
protocol:

```python
import math
import orbax.checkpoint.v1 as ocp
from typing import Sequence

class PrimeStepSavePolicy(ocp.training.save_decision_policies.SaveDecisionPolicy):
  def _is_prime(self, n):
    if n <= 1:
      return False
    for i in range(2, int(math.sqrt(n)) + 1):
      if n % i == 0:
        return False
    return True

  def should_save(
      self,
      step: ocp.CheckpointMetadata,
      previous_steps: Sequence[ocp.CheckpointMetadata],
      *,
      context: ocp.training.save_decision_policies.DecisionContext
  ) -> bool:
    # Save checkpoint if step number is prime.
    return self._is_prime(step.step)

# Usage
policy = PrimeStepSavePolicy()
ckptr = ocp.training.Checkpointer(directory, save_decision_policy=policy)
```

#### Aggregating Multiple Save Policies
Use `AnySavePolicy` to trigger a save if **any** of the underlying policies
evaluate to `True`.

```python
# Save every 1000 steps OR whenever a preemption is detected.
save_policy = ocp.training.save_decision_policies.AnySavePolicy([
    ocp.training.save_decision_policies.FixedIntervalPolicy(1000),
    ocp.training.save_decision_policies.PreemptionCheckpointingPolicy(),
])

ckptr = ocp.training.Checkpointer(directory, save_decision_policy=save_policy)
```

---

### 3.3 Mapped to PreservationPolicy
This section maps v0 options related to preservation decisions to their v1
equivalents. In v0, these options were used to determine which checkpoints to
keep. In v1, these concepts are now encapsulated within specific policies from
`ocp.training.preservation_policies`. These policies are passed directly via the
`preservation_policy` option during `Checkpointer` initialization. Multiple
preservation conditions, which in v0 were configured through various separate
options, can now be combined into a single `preservation_policy` using
`ocp.training.preservation_policies.AnyPreservationPolicy`.

| v0 Option | v1 Migration Logic | Description |
| :--- | :--- | :--- |
| `max_to_keep` | `LatestN(n=int)` | The number of most recent checkpoints to keep. <br> If `None`, all are preserved. |
| `keep_period` | `EveryNSteps(interval_steps=int)` | Ensures checkpoints are preserved every n steps. |
| `keep_time_interval` | `EveryNSeconds(interval_secs=int)` | Ensures checkpoints are preserved at least <br> after the specified time interval. |
| `should_keep_fn` | Implement a custom `PreservationPolicy`. | Predicate callable to determine survival of <br> a checkpoint. |
| `best_fn`, <br> `best_mode`, <br> `keep_checkpoints_without_metrics` | **BestN Configuration**: <br> `BestN(` <br> &nbsp;&nbsp;`get_metric_fn=Callable,` <br> &nbsp;&nbsp;`n=int,` <br> &nbsp;&nbsp;`reverse=bool,` <br> &nbsp;&nbsp;`keep_checkpoints_without_metrics=bool` <br> `)` | Maintains checkpoints based on quality metric <br> scores (higher/lower) rather than recency. <br> - `get_metric_fn`: Returns a scalar quality score. <br> - `n`: Map from `max_to_keep`. <br> - `reverse`: `True` for `min`, `False` for `max`. <br> - `keep_checkpoints_without_metrics`: If `False`, <br> checkpoints without metrics are eligible for cleanup. |
| `Multiple Preservation Rules` | `AnyPreservationPolicy([policy1, policy2])` | Preserves a checkpoint if any underlying <br> child policy suggests preservation. |

**Notes:**

**Shift to Explicit Global Preservation**:

In v0, configuration flags like `max_to_keep`, `keep_period`, and `best_fn` were
often **automatically combined** by the manager into a single composite logic.

In v1, providing a single policy (e.g., `LatestN(5)`) means **that is the only
preservation logic active**. If you want to keep the "latest 5" AND "every 1000
steps", you **must** explicitly use `AnyPreservationPolicy`. Providing
independent flags is no longer a supported pattern.

**v1 Migration**:

In v1, you must provide a `PreservationPolicy` directly to the `Checkpointer`
constructor. Legacy v0 options like `max_to_keep` or `keep_period` are no longer
supported as constructor arguments.

**Default Behavior**:

If no `preservation_policy` is provided, Orbax v1 defaults to
`ocp.training.preservation_policies.PreserveAll()`, which keeps every
checkpoint saved.

#### Custom PreservationPolicy Example
Implementing a custom preservation rule:

```python
import math
import orbax.checkpoint.v1 as ocp
from typing import Sequence

class PrimeStepPreservationPolicy(
    ocp.training.preservation_policies.PreservationPolicy
):
  def _is_prime(self, n):
    if n <= 1:
      return False
    for i in range(2, int(math.sqrt(n)) + 1):
      if n % i == 0:
        return False
    return True

  def should_preserve(
      self,
      checkpoints: Sequence[ocp.CheckpointMetadata],
      *,
      context: ocp.training.preservation_policies.PreservationContext
  ) -> Sequence[bool]:
    # Return a boolean for each checkpoint indicating survival.
    # We preserve checkpoints where step number is prime.
    return [self._is_prime(ckpt.step) for ckpt in checkpoints]

# Usage
policy = PrimeStepPreservationPolicy()
ckptr = ocp.training.Checkpointer(directory, preservation_policy=policy)
```

#### Aggregating Multiple Preservation Policies
Use `AnyPreservationPolicy` to ensure a checkpoint is kept if **any** of the
underlying policies suggest preservation (OR logic).

```python
# Keep the latest 5 checkpoints OR any checkpoint saved every 5000 steps.
preserve_policy = ocp.training.preservation_policies.AnyPreservationPolicy([
    ocp.training.preservation_policies.LatestN(5),
    ocp.training.preservation_policies.EveryNSteps(5000),
])

ckptr = ocp.training.Checkpointer(
    directory, preservation_policy=preserve_policy
)
```

---

### 3.4 Mapped to step_name_format
Naming formats are imported from `ocp.path.step`.

| v0 Option | v1 Migration Logic | Description |
| :--- | :--- | :--- |
| `step_prefix`, <br> `step_format_fixed_length`, <br> `single_host_load_and_broadcast` | **standard_name_format Configuration**: <br> `standard_name_format(` <br> &nbsp;&nbsp;`step_prefix=str,` <br> &nbsp;&nbsp;`step_format_fixed_length=int,` <br> &nbsp;&nbsp;`single_host_load_and_broadcast=bool` <br> `)` | Defines naming convention for step directories. <br> - `step_prefix`: simple prefix for checkpoint naming (e.g., `test_0`). <br> - `step_format_fixed_length`: Defines length of zero padding (e.g., `000123`). <br> - `single_host_load_and_broadcast`: If `True`, <br> process 0 lists steps and broadcasts to others. |

**v1 Migration**:

In v1, all naming configuration is encapsulated in the `step_name_format`
object. Legacy v0 strings like `step_prefix` are no longer passed as top-level
arguments to the `Checkpointer`.

**Default Behavior**:

If no `step_name_format` is provided, Orbax v1 defaults to
`ocp.path.step.standard_name_format()`, which uses simple integer directory
names (e.g., `0`, `1`, `100`).

#### Aggregating Naming Formats (Composite)
The `composite_name_format` allows you to support multiple reading formats
while adhering to a single format for writing.

```python
# Allow reading 'checkpoint_100' or '100', but always write as 'checkpoint_100'.
name_format = ocp.path.step.composite_name_format(
    write_name_format=ocp.path.step.standard_name_format(
        step_prefix='checkpoint'
    ),
    read_name_formats=[
        ocp.path.step.standard_name_format(step_prefix='checkpoint'),
        ocp.path.step.standard_name_format(step_prefix=None),
    ]
)

ckptr = ocp.training.Checkpointer(directory, step_name_format=name_format)
```

---

### 3.5 Managed via Orbax Context
These environmental settings are set once in the `ocp.Context`.

Note: This table only covers v0 options and isn't comprehensive of all context
options and fields available for configuration, please see
`orbax.checkpoint.v1.context.options` for further detail.

| v0 Option | v1 Equivalent | Context Path | Description |
| :--- | :--- | :--- | :--- |
| `async_options` | `AsyncOptions` | `Context.async_options` | Configure properties of async behavior. <br> **Note**: `barrier_sync_fn` is removed. |
| `multiprocessing_options` | `MultiprocessingOptions` | `Context.multiprocessing_options` | Configures multiprocessing behavior <br> (e.g., primary host, barrier prefix). |
| `file_options` | `FileOptions` | `Context.file_options` | Options to configure checkpoint directories <br> and files. Defaults to `FileOptions()`. |
| `todelete_full_path` | `todelete_full_path` | `Context.deletion_options.gcs_deletion_options.todelete_full_path` | Path for "soft-deleting" GCS checkpoints. |


#### Granular Comparison: File & Multiprocessing Options

**Structural Mismatch in v1 Options**

v1 options classes (`FileOptions`, `MultiprocessingOptions`, `AsyncOptions`)
are designed with a stricter API:

`v1.FileOptions` is a fresh implementation. Unlike `AsyncOptions` which is an
extension, `FileOptions` has a different internal structure.

**MultiprocessingOptions**

*   **Removal of `barrier_sync_fn`**: Orbax now manages process synchronization
    automatically using `jax.distributed_client` or thread-safe local fallback.

**FileOptions**

*   **Definition**: `v1.FileOptions` focuses on environmental/I/O settings
    rather than training lifecycle flags.

*   **Added in v1**:
    *   `path_class`: Ability to override the `epath` implementation.

While v1 options provide a `.v0()` method for internal conversion, users
should manually reconstruct their `v1.FileOptions` to ensure that flags are
correctly configured within the `Context`.

---

### 3.6 Defaulted and Unsupported v0 Configuration Options
The following options have either been made default or are no longer supported.

| v0 Option | V1 Behavior |
| :--- | :--- |
| `create` | Directory creation is now enabled by default. Orbax will <br> attempt to create the checkpoint directory if it does not <br> already exist. |
| `save_root_metadata` | Root metadata management is now internalized by default. |
| `prevent_write_metrics` | Metrics persistence is managed to ensure integrity. |
| `check_storage_locality` | Always checks and logs network locality. |
| `enable_background_delete`| Deletion is managed internally; no longer configurable. |
| `read_only` | Behavior unsupported entirely. |
| `enable_per_process_directory_creation` | Behavior unsupported entirely. |
| `enable_should_save_is_saving_in_progress_check` | Behavior unsupported entirely. |
| `temporary_path_class` | Unsupported. Orbax now automatically determines the correct atomicity protocol based on the filesystem backend to prevent misconfiguration. |

---

#### 3.7 Explicit Async Control

| v0 Option | v1 Equivalent |
| :--- | :--- |
| `enable_async_checkpointing` | Manual Delegation |

In v1, we avoid ambiguous "global" async flags. If you previously set
`enable_async_checkpointing=True`, you should now use the `save_*_async`
methods, which return an `AsyncResponse` object on which you can call
`.result()` to block until completion. If you previously set
`enable_async_checkpointing=False`, you should use the synchronous `save_*`.

---

## 4. Common Migration Pitfalls

- v1 options (like `MultiprocessingOptions`) are `kw_only=True`. Use:
  `MultiprocessingOptions(primary_host=0)` instead of
  `MultiprocessingOptions(0)`.
- **Modifying Immutable Options**:
  v1 options are `frozen=True`. Use
  `dataclasses.replace(options, timeout_secs=100)`.
- **Forgotten Preservation Policy**:
  Failing to pass one to v1 `Checkpointer` results in unlimited growth.

---

## 5. Best Practices for v1 Configuration

1. **Use `ocp.Context` for Environment-Level Settings**:
   Set `AsyncOptions` or `MultiprocessingOptions` once at your task entry
   point.
2. **Explicitly Manage Policies**:
   Always pass explicit `SaveDecisionPolicy` and `PreservationPolicy` objects.
3. **Prefer Context Manager for `Checkpointer`**:
   Use `with ocp.training.Checkpointer(directory) as ckptr:` to ensure
   `close()`.
