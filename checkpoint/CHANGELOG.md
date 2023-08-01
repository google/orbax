# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2022-08-01

### Added
- Fully self-describing PyTree checkpoints with type information, stored in
metadata using JSON format (not currently enabled by default).
- PyTree leaf metadata returned by `metadata` function (dependent on the 
above.)

### Fixed
- Correctly set TYPESTR_REGISTRY, to account for OCDBT option.

### Removed
- Deprecated `restore_args_from_target` function.


## [0.3.1] - 2022-07-27

### Added
- Option to not port `global_shape` to `ArrayRestoreArgs`.

### Removed
- Protobuf metadata saved by PyTreeCheckpointHandler.

## [0.3.0] - 2022-07-24

### Added
- `close` method for Checkpointer and CheckpointHandler. 
- Context manager helper functions for Checkpointer and CheckpointManager.
- Protobuf metadata saved by PyTreeCheckpointHandler.

### Changed
- Allow calling `create_coordinator_server_and_context` without an initialized
JAX coordinator server and create TS metadata for numpy arrays on a single process.
- Refactor TypeHandler to operate over batches of values, rather than individual ones.
- Removed support for lazy restoration. Supported via transformations.

## [0.2.6] - 2022-06-28

### Added
- ProtoCheckpointHandler.

### Fixed
- Fix issue with `multi_value_fn` in restoration where an input value could be
loaded many times unnecessarily.
- Eliminates hosts sync on background thread and fixes issue with reading
lockfile before checkpoint is finalized.
- Fix unlocking function, which may fail if there are multiple evaluators
running concurrently.

## [0.2.6] - 2022-06-16

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

## [0.2.5] - 2022-06-09

### Added
- Support `max_to_keep=0`.
- Support for PyTree keys with '/' (à la Haiku).

### Fixed
- GCS error when `cleanup_tmp_directories=False` which caused an internal
assertion to be raised when saving over an existing temporary directory.

## [0.2.4] - 2022-06-01

### Added

- `merge_trees` function in `transform_utils`.

### Changed
- Explicit Python version support for 3.9, 3.10, 3.11
- Raise ValueError when trying to save jax.Array to the aggregate file if it is
not fully replicated.


## [0.2.3] - 2022-05-12

### Added

- Raise error message when the user tries to save host local arrays that are
typically obtained using pmap.

## [0.2.2] - 2022-05-08

### Added
- Option to allow users to disable automatic temporary directory cleanup upon
CheckpointManager initialization.
- Error message when metadata file ('.zarray') is missing.
- `reached_preemption` function to allow the user to detect if a preemption signal has been received.

### Changed
- Set `create` option to True by default.

### Fixed
- Msgpack encoding of tuples.

## [0.2.1] - 2022-04-14

### Fixed
- Asyncio issue affecting python<=3.9.

## [0.2.0] - 2022-04-10

### Added
- Tensorstore options to improve OCDBT performance.
- Add support for `value_fn` transformations during restore.
- Support for `multi_value_fn` transformations during restore.

### Changed
- Reworked transformations logic in restore to happen in a more intuitive order,
with lazy loading to avoid materializing unnecessary arrays.

### Fixed
- Slow repeated calls to check whether a checkpoint is OCDBT format or not.

## [0.1.8] - 2022-03-31

### Changed
- Increased minimum tensorstore version to what's needed for OCDBT.

## [0.1.7] - 2022-03-28

### Added
- `orbax-checkpoint` is introduced, a namespace package under `orbax`. Importing
this package takes the form `import orbax.checkpoint` or 'from orbax import
checkpoint`.
- Support for OCDBT driver in Tensorstore.

## [0.1.6] - 2022-03-22

### Fixed
- Small bug fixes.

## [0.1.5] - 2022-03-17

### Added
- Use a more precise timestamp when generating temporary directory names to
permit more than one concurrent checkpointing attempt per second.
- Automatic import of nest_asyncio.

## [0.1.4] - 2022-03-15

### Added
- Support for generic transformation function in PyTreeCheckpointHandler.
- Support n-digit checkpoint step format.

### Fixed
- Eliminate Flax dependency to fix circular dependency problem.

## [0.1.3] - 2022-03-03

### Added
- `sharding` option on `ArrayRestoreArgs

## [0.1.2] - 2022-02-17

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

## [0.1.1] - 2022-01-30

### Added
- A util function that constructs restore_args from a target PyTree.
- CheckpointManager `delete` API, which allows deleting an existing step.
- Made dev dependencies optional to minimize import overhead.

### Changed
- Refactored higher-level utils in checkpoint_utils, which provides
user-convenience functions.
- Guard option to create top-level directory behind `create` option.
- Remove support for Python 3.7.

## [0.1.0] - 2022-01-04

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

