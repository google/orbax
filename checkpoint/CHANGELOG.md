# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Adjust user metadata construction so that older checkpoints with some values
in the msgpack file can still return metadata (though the metadata for these
entries will not return information about array properties).

## [0.5.21] - 2024-07-12

### Added
- Rolled forward change to improve TensorStore I/O efficiency.
- Memory efficient broadcasting from one model replica to others.
- Ability to check if a checkpoint save is in progress.

### Changed
- Allow one directory creation request per item rather than 1 per item per host.
- Make atomicity logic configurable, and encapsulate it within a class.

### Fixed
- Refactor ts.Context usage to be per-operation (save/restore) rather than a
  global object. This helps fix edge cases where a user repeatedly overwrites
  a single checkpoint.
- Move module-level counters to one place for greater compartmentalization.
  Add logic to the barrier-compatible test fixture that allows each test case to
  have its own module-level counter, to avoid problems that arise when
  multiprocess tests run in inconsistent orders.
- Ensure D2H transfers are parallelized.

## [0.5.20] - 2024-06-20

### Added
- Fork JAX serialization library into orbax/checkpoint.

## [0.5.19] - 2024-06-20

### Fixed
- updated required Jax version to fix PY3.9 build

## [0.5.18] - 2024-06-20

### Fixed

- Earlier change relying on a flag that was not available in all environments.

## [0.5.17] - 2024-06-18

### Fixed
- Rolled back change in previous release to improve TensorStore I/O efficiency.
  This change caused some unexpected failures on certain storage systems.

## [0.5.16] - 2024-06-11

### Added
- Checkpoint format guide for RTD page.

### Changed
- Modify `_write_metadata_file` to Async.
- Added `tree` package to contain tree-related utilities.
- Improve TensorStore I/O efficiency through use of TensorStore
  transactions for the OCDBT storage format, and specify the new
  `can_reference_source_data_indefinitely=True` option to avoid a redundant copy
  when writing into the TensorStore chunk cache.
- Stop writing msgpack file for new checkpoints and update empty nodes handling
so that it no longer depends on this file.

## [0.5.15] - 2024-05-31

### Added
- Introduce `FileOptions` as CheckpointManagerOptions attribute.
- Support non blocking CheckpointMetadataStore.write.

### Fixed
- Deadlock observed when using multiple AsyncCheckpointers at once.

## [0.5.14] - 2024-05-23

### Changed
- Delegate to BasePyTreeCheckpointHandler rather than inheriting from it.

## [0.5.13] - 2024-05-23

### Changed
- Emergency checkpointing bug-fixes

## [0.5.12] - 2024-05-23

### Added
- Introduce `should_save_fn` in `OrbaxCheckpointManagerOptions`.
- Introduce `StepAlreadyExistsError` to be raised on save with existing step.

### Changed
- Modify `JsonCheckpointHandler` to Async.

### Fixed
- Fix empty metadata file error: Expecting value: line 1 column 1 (char 0)


## [0.5.11] - 2024-05-10

### Added
- Implement restoration in emergency.CheckpointManager.
- Separate `metadata` package to encapsulate metadata-related utils and
constructs.
- Add step utils to get metadata of latest checkpoint and from checkpoint path.
- Support checkpoint metadata at step level.
- Use checkpoint metadata module to store commit timestamp.

### Changed
- In `checkpoints_iterator`/`wait_for_new_checkpoint`, ensure that if steps are
present, they will be yielded even if `timeout_fn` already returns True.
- Refactor and move path and step utils to new path/utils and step modules
 respectively.
- Refactor Tensorstore-related codes in type_handlers.py.
- Update `NameFormat.find_step` logic to exclude uncommitted checkpoints.
- Abstract `jax.process_index` into `multihost.process_index`.
- Factor out core PyTree checkpointing logic into a
`PyTreeCheckpointHandlerImpl` class.
- Use unique count instead of timestamp in tmp directory construction.
- Tidy up `NameFormat` API: remove `build_metadata` and `find_metadata` methods
  from the public API.

### Removed
- `ocdbt_merge` option and unused `restore_with_serialized_types` option from
`PyTreeCheckpointHandler`.
- OCDBT coordinator code. These functions are no longer needed.
- `write_tree_metadata` option, as there is no real reason to disable this now.

## [0.5.10] - 2024-04-22

### Added
- Add path package to export symbols, also add Step rst docs.
- Create composite step `NameFormat`.
- Docs on working with PyTrees/arrays.

### Changed
- Improve step lookup error message by adding expected names to it.
- Error messages when sharding is not specified.
- For `broadcast_one_replica_to_all`, delete input arrays as soon as possible to
conserve memory.

### Fixed
- Added timeout_fn arg to `wait_for_new_checkpoint` and
`_wait_for_new_checkpoint`

## [0.5.9] - 2024-04-02

### Fixed
- Support for Python 3.9

## [0.5.8] - 2024-04-02

## Added
- Add CheckpointManagerOptions.enable_background_delete to avoid blocking
  the manager.save() code path
- `broadcast_one_to_some` function.
- Allow running Orbax code on a subset of processes.
Note that this currently does not work on async code.
- Add `SingleReplicaArrayRestoreArgs.primary_replica_index` to select which
replica to load checkpoint and broadcast when `SingleReplicaArrayHandler` is
used.

### Changed
- CheckpointManager is defined as a ContextManager directly
- `checkpoint_manager_context` is deprecated
- Checkpointer is defined as a ContextManager directly
- `checkpointer_context` is deprecated
- AsyncCheckpointer is defined as a ContextManager directly
- `async_checkpointer_context` is deprecated
- Refactored to create `multihost_utils` module.
- Remove unnecessary barriers in CheckpointHandlers.

## [0.5.7] - 2024-03-20

### Added
- Added Zarr3 support for numpy array
- Added PyTreeSaveArgs.ocdbt_target_data_file_size to control the
  target_data_file_size when OCDBT is enabled
- Expanded skeleton of `emergency.CheckpointManager`.

### Fixed
- Fix TypeHandler Registry errors for `empty ([], {}, None)` values with
 `SaveArgs(aggregate=False)`.

## [0.5.6] - 2024-03-14

### Added
- Add new `step` module with step naming and query support.
- Add new option `enable_background_delete` to CheckpointManager so that
 old checkpoints can be deleted in the background.

### Changed
- Use step `NameFormat` and `StandardNameFormat` in Orbax `CheckpointManager`,
 `utils` and `checkpoint_utils` modules.

## [0.5.5] - 2024-03-11

## Fixed
- Add JAX version guards to changes that require newer version of JAX

## [0.5.4] - 2024-03-08

### Fixed
- Issue in which `CompositeCheckpointHandler` would create item directories
repeatedly in parallel, resulting in multiple requests per host, for every host.

### Changed
- Log `SaveArgs.aggregate` deprecation warning message once in 12 hours.
- If CheckpointManagerOptions.read_only=True then automatically reset "write"
 options (instead of raising error).
- Modify `create_empty` to be `erase_and_create_empty` instead to make its
potential dangers a bit more apparent.

### Added
- Single replica broadcasting when training on multiple hosts/pods. By default,
replica zero along the first axis dimension reads the checkpoint then broadcasts
to other replicas.
- Added experimental_emergency_checkpoint
- `ShardingMetadata` class that represents `jax.sharding.Sharding` properties but does not require accessing real devices.

## [0.5.3] - 2024-02-05

### Fixed
- Fix broken logging issue.

### Added
- Add `reload` method in `CheckpointManager` which resets internal properties.
- Add checkpoint announcements RTD page.

### Changed
- Deprecate `read` option in `all_steps`.
- Deprecate `SaveArgs.aggregate`.
- Update copyright to use 2024. Also, allow notebook cells to continue if last
 cell raises error.
- Improve RTD by using `ocp.utils.to_shape_dtype_struct` instead of `jax.eval_shape`.

## [0.5.2] - 2024-01-26

### Fixed
- Ensure timeout passed via `AsyncCheckpointer` in `CheckpointManager` is
propagated, for legacy compatibility.
- Modified sharding file writes to use `tensorstore.Transaction` due to recent
tensorstore change. This resolves a slowdown in save speed recently observed.

### Added
- Add documentation associated with `item_handlers`, the new arguments
 introduced in `CheckpointManager` constructor.

### Changed
- Update documentation of the new `CheckpointArgs` based `CheckpointHandler` API.


## [0.5.1] - 2024-01-19

### Fixed
- Stop blocking on previous save when `should_save` is False.

### Added
- Expose `AsyncOptions` for `CheckpointManager` users.
- Introduce item_handlers to CheckpointManager ctor to allow configurable
 Handler setup.
- Add JaxRandomKeyCheckpointHandler to store Jax random key generated from
jax.random.key() or jax.random.PRNGKey()
- Add NumpyRandomKeyCheckpointHandler to store Numpy random state from
numpy.random.get_state()

## [0.5.0] - 2024-01-16

### Added
- Deprecation warning for users of the old `CheckpointManager` API.
- `CheckpointManager` API migration guide.
- New backwards-compatible API for `CheckpointManager` making use of
`CheckpointArgs` and `CompositeCheckpointHandler`.
- Documentation for new APIs.

### Changed
- Provide better support for custom `CheckpointHandler`s without registered
`CheckpointArgs` by providing a wrapper `CheckpointHandler` as a fallback. This
class is introduced for backwards compatibility, and will eventually be removed.

## [0.4.8] - 2023-12-14

### Added
- New parameter `chunk_byte_size` in `SaveArgs`.  A convenient way to choose
the write and read chunk shapes using Zarr3.

### Fixed
- Issues interpreting `ValueMetadata` in `construct_restore_args`.

## [0.4.7] - 2023-12-07

### Fixed
- Refactor bad `wait_until_finished` design in `CheckpointManager` where
`wait_until_finished` would try to join a thread, which itself called
`wait_until_finished`.

## [0.4.6] - 2023-12-06

### Added
- `CompositeCheckpointHandler` API. This will soon replace `CheckpointManager`'s
handling of distinct items, and allows users to work with separate items at the
`Checkpointer` layer.

### Changed
- `use_ocdbt` `PyTreeCheckpointHandler` option is no longer kept as a mutable
global state on type handlers and is now passed around with the state to save or
restore.

### Fixed
- Bug arising when an error occurs in the background thread, and we try to
remove a non-existent checkpoint from `_interval_preserved_checkpoints` in
`CheckpointManager`.

## [0.4.5] - 2023-12-04

### Added
- Zarr Version 3 allowing custom write and read chunk configurations

## [0.4.4] - 2023-11-29

### Fixed
- Bug where errors in the background thread of `AsyncCheckpointer` would not be
raised in `CheckpointManager`, causing later errors when trying to remove non-
existent checkpoints.

### Added
- Support for `CheckpointArgs` in core `CheckpointHandler` implementations.
Allowed specifying either `CheckpointArgs` or keyword args in `Checkpointer`.
- Introduce CheckpointManagerOptions.read_only to control save/delete behaviors.

### Changed
- Use `json` directly instead of `JsonCheckpointHandler` to write and read
`PyTreeCheckpointHandler`'s metadata.
- Enable OCDBT-Merge by default

## [0.4.3] - 2023-11-17

### Added
- Introduce CheckpointManagerOptions.todelete_subdir option to rename deletable dirs.
- Support jax.sharding.SingleDeviceSharding in self-describing PyTree checkpoints.
- `name` and `directory` properties in value `Metadata`.
- Introduce AbstractCheckpointManager protocol for the already existing
 CheckpointManager concrete class.

### Changed
- Turn on self-describing PyTree checkpoints by default.
- `CheckpointArgs` arguments for save and restore in `StandardCheckpointHandler`
and `PyTreeCheckpointHandler`.
- Return empty dict if the CheckpointManager level metadata is not available.
Currently it raises error due to missing metadata dir.
- Remove unfinalized checkpoint directory from previous runs for GCS.

## [0.4.2] - 2023-10-03

### Added
- Custom `finalize` callback for `CheckpointHandler`.
- Merge/finalize logic for Tensorstore when using OCDBT driver.

### Changed
- Barrier synchronization in `AsyncCheckpointer` refactored to allow plugging in
alternative implementations.
- Added `parent_dir` under `ParamInfo` and modified sharding file to be saved
under `ParamInfo.parent_dir`.
- In `all_steps`, minimize disk queries on non-leader processes.
- Put `all_steps` load on single host and broadcast feature behind an option that defaults to False.
- Rename StandardCheckpointHandler.save/restore named arg, 'state' to 'item'.

### Removed
- OCDBT coordinator.

## [0.4.1] - 2023-09-28

### Added
- `CompositeCheckpointHandler`
- `CheckpointArgs`

### Changed
- Forked `AsyncManager` into Orbax and replaced `AsyncCheckpointer`'s
inheritance from it with composition for easier customization.

## [0.4.0] - 2023-09-20

### Added
- Added support for automatically inferring sharding if not provided by
`RestoreArgs`.

### Fixed
- Fix sync error with removing old checkpoints.
- Fix missing checkpoint benchmarks images in RTD page.
https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html#introducing-ocdbt

### Changed
- Use `nest_asyncio` by default. This allows users to make calls to orbax from
within `async` functions.
- Modified `StringHandler` serialization and deserialization to use Tensorstore
Json driver for async file reads and writes.
- Marked `transform_utils` as deprecated.
- Change `PytreeCheckpointHandler`'s parameter `use_ocdbt` default to `True`
- Modified sharding property in value_metadata to
`jax.sharding.Sharding`

## [0.3.5] - 2023-08-17

### Fixed
- User metadata when dealing with empty nodes that follow non-empty nodes in a
list.

### Changed
- Modify `_get_user_metadata` to exclude empty nodes.

### Fixed
- Fix GCS issue where an error would be encountered when trying to save a step
with the same number as an existing tmp directory. This scenario arises when
restarting after preemption, without enabling `cleanup_tmp_directories`
in `CheckpointManager`.
- Fix `create_coordinator_server_and_context()` breaking old codes that expect it to return a tuple.  The function also prints a deprecation warning.

## [0.3.4] - 2023-08-15

### Added
- `StandardCheckpointHandler`.

## [0.3.3] - 2023-08-12

### Fixed
- Fix concurrency issue when `are_locked` check lags behind checkpoint deletion.

## [0.3.2] - 2023-08-01

### Added
- Fully self-describing PyTree checkpoints with type information, stored in
metadata using JSON format (not currently enabled by default).
- PyTree leaf metadata returned by `metadata` function (dependent on the
above.)

### Fixed
- Correctly set TYPESTR_REGISTRY, to account for OCDBT option.

### Removed
- Deprecated `restore_args_from_target` function.


## [0.3.1] - 2023-07-27

### Added
- Option to not port `global_shape` to `ArrayRestoreArgs`.

### Removed
- Protobuf metadata saved by PyTreeCheckpointHandler.

## [0.3.0] - 2023-07-24

### Added
- `close` method for Checkpointer and CheckpointHandler.
- Context manager helper functions for Checkpointer and CheckpointManager.
- Protobuf metadata saved by PyTreeCheckpointHandler.

### Changed
- Allow calling `create_coordinator_server_and_context` without an initialized
JAX coordinator server and create TS metadata for numpy arrays on a single process.
- Refactor TypeHandler to operate over batches of values, rather than individual ones.
- Removed support for lazy restoration. Supported via transformations.

## [0.2.6] - 2023-06-28

### Added
- ProtoCheckpointHandler.

### Fixed
- Fix issue with `multi_value_fn` in restoration where an input value could be
loaded many times unnecessarily.
- Eliminates hosts sync on background thread and fixes issue with reading
lockfile before checkpoint is finalized.
- Fix unlocking function, which may fail if there are multiple evaluators
running concurrently.

## [0.2.6] - 2023-06-16

### Added
- Locking mechanism for `checkpoints_iterator` to prevent checkpoints from
being cleaned up by a `CheckpointManager` in another process while the are being
read.
- Allow saving the aggregated file asynchronously.

### Changed
- Slightly different behavior for `wait_for_new_checkpoint`, allowing it to
wait until a certain checkpoint, rather than strictly after a given step.
- Allow value_fn and multi_value_fn to accept RestoreArgs as an argument when they are used during restore, so that the user may customize the returned value based on what is requested by RestoreArgs.

### Fixed
- Support creating sharded array when ArrayRestoreArgs is passed and the value
was originally aggregated.

## [0.2.5] - 2023-06-09

### Added
- Support `max_to_keep=0`.
- Support for PyTree keys with '/' (à la Haiku).

### Fixed
- GCS error when `cleanup_tmp_directories=False` which caused an internal
assertion to be raised when saving over an existing temporary directory.

## [0.2.4] - 2023-06-01

### Added

- `merge_trees` function in `transform_utils`.

### Changed
- Explicit Python version support for 3.9, 3.10, 3.11
- Raise ValueError when trying to save jax.Array to the aggregate file if it is
not fully replicated.


## [0.2.3] - 2023-05-12

### Added

- Raise error message when the user tries to save host local arrays that are
typically obtained using pmap.

## [0.2.2] - 2023-05-08

### Added
- Option to allow users to disable automatic temporary directory cleanup upon
CheckpointManager initialization.
- Error message when metadata file ('.zarray') is missing.
- `reached_preemption` function to allow the user to detect if a preemption signal has been received.

### Changed
- Set `create` option to True by default.

### Fixed
- Msgpack encoding of tuples.

## [0.2.1] - 2023-04-14

### Fixed
- Asyncio issue affecting python<=3.9.

## [0.2.0] - 2023-04-10

### Added
- Tensorstore options to improve OCDBT performance.
- Add support for `value_fn` transformations during restore.
- Support for `multi_value_fn` transformations during restore.

### Changed
- Reworked transformations logic in restore to happen in a more intuitive order,
with lazy loading to avoid materializing unnecessary arrays.

### Fixed
- Slow repeated calls to check whether a checkpoint is OCDBT format or not.

## [0.1.8] - 2023-03-31

### Changed
- Increased minimum tensorstore version to what's needed for OCDBT.

## [0.1.7] - 2023-03-28

### Added
- `orbax-checkpoint` is introduced, a namespace package under `orbax`. Importing
this package takes the form `import orbax.checkpoint` or 'from orbax import
checkpoint`.
- Support for OCDBT driver in Tensorstore.

## [0.1.6] - 2023-03-22

### Fixed
- Small bug fixes.

## [0.1.5] - 2023-03-17

### Added
- Use a more precise timestamp when generating temporary directory names to
permit more than one concurrent checkpointing attempt per second.
- Automatic import of nest_asyncio.

## [0.1.4] - 2023-03-15

### Added
- Support for generic transformation function in PyTreeCheckpointHandler.
- Support n-digit checkpoint step format.

### Fixed
- Eliminate Flax dependency to fix circular dependency problem.

## [0.1.3] - 2023-03-03

### Added
- `sharding` option on `ArrayRestoreArgs

## [0.1.2] - 2023-02-17

### Added
- Add "standard user recipe" to documentation.
- Add unit tests using mock to simulate preemption.
- Logging to increase transparency around why checkpoints are kept vs. deleted.
- Expand on uses of restore_args in colab.
- Expose utils_test.
- Add msgpack_utils to move toward eliminating Flax dependency.
- CheckpointManager starts a background thread to finalize checkpoints so that
checkpoints are finalized as soon as possible in async case.

### Changed
- Remove CheckpointManager update API.
- Remove support for deprecated GDA.
- Add tmp suffix on step directory creation in CheckpointManager.save.

### Fixed
- Preemption when using keep_time_interval caused the most recent steps before
preemption to be kept, despite not falling on the keep time interval.

## [0.1.1] - 2023-01-30

### Added
- A util function that constructs restore_args from a target PyTree.
- CheckpointManager `delete` API, which allows deleting an existing step.
- Made dev dependencies optional to minimize import overhead.

### Changed
- Refactored higher-level utils in checkpoint_utils, which provides
user-convenience functions.
- Guard option to create top-level directory behind `create` option.
- Remove support for Python 3.7.

## [0.1.0] - 2023-01-04

### Added

- Check for metric file in addition to item directory in CheckpointManager.
- Additional logs to indicate save/restore completion.
- Support for None leaves in PyTree save/restore.
- ArrayCheckpointHandler for individual arrays/scalars.
- `read: bool` option on all_steps to force read from storage location instead
of using cached steps.
- Simplified "Getting Started" section in the docs.
- CheckpointManager creates the top level directory if it does not yet exist.
- Write msgpack bytes asynchronously.

### Changed
- Removed some unused test_utils methods for filtering empty nodes.
- Update docs on `PyTreeCheckpointHandler`.
- Removed unneeded AbstractCheckpointManager.

### Fixed

- Usage of bytes_limiter to prevent too many bytes from being read during a
single restore call.
- Temp checkpoint cleanup when using a step prefix (i.e. 'checkpoint_0').

## [0.0.23] - 2022-12-08

### Added

- Option to customize metadata file name for Tensorstore.

### Fixed

- Restore failure on GCS due to misidentification of checkpoint as
"not finalized".

## [0.0.22] - 2022-12-05

### Added

- Added CHANGELOG.md for version updates (additions and changes), ingested by
auto-publish functionality.

## [0.0.21] - 2022-12-05

### Changed

- Fix mistaken usages of placeholder "AGGREGATED" where "NOT-AGGREGATED" would
be more appropriate. Ensure backwards compatibility is maintained.
