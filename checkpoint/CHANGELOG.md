# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.30] - 2025-11-26

### Fixed

- Roll back earlier change altering metadata format, which was observed to cause
breakages.

## [0.11.29] - 2025-11-25

### Fixed

- Fix `step_from_checkpoint_name` to allow the passed in checkpoint name to
  include an arbitrary `step_prefix` with any character(s) such as underscores.
- Fix CheckpointManager initial directory creation to use `file_options.path_permission_mode`.
- Fix using jax.eval_shape with StandardRestore
- Fix checkpoint restoration logic for integer dict keys. Ensures
  these keys, which were being temporarily stored as strings, are properly
  restored back to integers.

### Changed

- Validate checkpoints before writing merged OCDBT database using in-memory
  state, avoiding additional I/O to re-read metadata.
- add `support_format` to utils.to_shape_dtype_struct()
- Moved `register_pathways_handlers` to `ocp.pathways.register_type_handlers`.
- Replace usage of `get_json_tpec_read` and delegate functionality to new
  function `build_array_read_spec` which constructs and returns an
  `ArrayReadSpec`.

## [0.11.28] - 2025-11-06

### Added

- #v1 `Checkpointer.wait` method.
- Added `register_pathways_handlers` to `ocp.type_handlers`

## [0.11.27] - 2025-11-04

### Added

- #v1 Add Save option `enable_replica_parallel_separate_folder`
- #v1 Add Save option `min_slice_bytes_for_replica_parallel` and `max_replicas_for_replica_parallel`
- Add support for dispatchers in Jax Array Handlers.
- Add support to restore JAX random key with abstract representation such as
  `ShapeDtypeStruct`

### Changed

- #v1 Allow PyTree checkpointables to be automatically identified in
both `pytree_metadata` and `load_pytree`.

## [0.11.26] - 2025-10-22

### Added

- Added an option `use_compression` to `PyTreeCheckpointHandler` to
  control whether compression will be used. Default is kept `True`.
- v1 Added `ArrayOptions.Saving.use_compression` to control whether compression
  will be used. Default is kept `True`.
- Add `LocalPath` for testing process-local storage.
- Added new memory metrics into benchmark library
- Added `Dispatcher` interface with `ColocatedPythonDispatcher` implementations,
  enabling function execution on colocated CPU devices.
- Add `enable_replica_parallel_separate_folder` into type_handlers.ArrayHandler
  which will store replica data in separate folders.

### Changed

- #v1 Modify LeafHandler definitions so that `AbstractLeaf` or
`Type[AbstractLeaf]` are always accepted as valid abstract values.

## [0.11.25] - 2025-09-10

### Changed

- Move temporary path detection into `TemporaryPath` class. This class serves
as a natural place for encapsulating core temporary path detection logic.
- Simplify `find_step_path` logic by removing special casing for different
filesystems - instead depending on `TemporaryPath` to identify uncommitted
paths.
- Remove the `jax_pmap_no_rank_reduction=False` code path for
`utils.fully_replicated_host_local_array_to_global_array`.
`jax_pmap_no_rank_reduction=True` by default in JAX.
- Changed to minimum jax version requirement to v0.6.0

## [0.11.24] - 2025-08-28

### Added

- Add support to enable orbax internal metadata inside root metadata.

### Changed

- Broadcast max steps as well when using single_host_load_and_broadcast.
- Allow changing tensorstore backend through environment variable
  `TENSORSTORE_GCS_BACKEND`. The default is `gcs`. Utilizing `gcs_grpc` instead
  provides roughly double the GCS performance.

## [0.11.23] - 2025-08-18

### Added

- Add support for sharded loading from Safetensors checkpoints onto a JAX mesh.
- Ability to have unique per process signalling for directory creation.

## [0.11.22] - 2025-08-14

### Changed

- Allow ArrayMetadataStore to update existing metadatas.

### Added

- Orbax Emergency Checkpointer will now do single slice restore from persistent storage.

## [0.11.21] - 2025-08-07

### Fixed

- Fix incorrectly specified timeouts waiting for signals in background threads.
Decrease timeout to 60s.

### Changed

- #cleanup De-duplicate scattered implementations of async Path functions and
centralize definitions.

### Added

 - Add `metadata()` method for loading with SafeTensors checkpoint format.
Refactored `metadata/loading.py` to allow this change.
- Add options to control number of replicas and min bytes in replica-parallel.

## [0.11.20] - 2025-07-28

### Added

- Add support for loading SafeTensors checkpoints
- #v1 Add `is_orbax_checkpoint()` method for validation checks
- #v1 Add custom LeafHandler docs.
- #v1 Add leaves and abstract leaves validation before saving and loading.

### Changed

- Modify `loading.load_pytree()` method to automatically detect and handle
different formats
- #v1 Adjust v0/v1 compatibility with CheckpointLayout loading logic.
Require users to explicitly configure a layout if it is not Orbax.
- #v1 Update all V1 Codes work with V1 (De)SerializationParams and Metadata.
  Avoid mixing with V0 ones.
- #v1 Add LeafHandlerRegistry and make it as default for V1 codes
- #v1 Update LeafHandlerRegistry to work handler type only.

## [0.11.19] - 2025-07-08

### Added

- #v1 Add dummy `orbax.checkpoint` file to help identify checkpoints.
- #v1 Add StringLeafHandler

### Changed

- Additional fixes and temporary options for Cloud users.

## [0.11.18] - 2025-07-01

### Changed

- For JAX>=0.6.2, JAX layout.Layout renamed to layout.Format

## [0.11.17] - 2025-06-30

### Added

- #v1 Add `PreservationPolicy` support in Checkpointer.
- Fix rmtree behavior for GCS HNS enabled buckets.
- #v1 Update AbstractScalar as a simple `Type`

### Changed

- #v1 remove unnecessary warnings when leaf handlers are added into existing
type handler registry

## [0.11.16] - 2025-06-18

### Changed

- make sure the GitHub build.yml allow to install newer jax-version
- #v1 Modify saving code to eliminate cross-thread event loop sharing.

## [0.11.15] - 2025-06-13

### Changed

- #emergency-checkpoint Stop including barrier processes in barrier names. This
information should be redundant, since the barrier ID always corresponds 1:1
with the provided processes ('local_all_steps':
processes_excepting_primary_slice, 'find_complete_slice`: all_processes).
- #emergency-checkpoint Support a `keep_period` like the regular
`CheckpointManager`.
- #v1 Default using V1 LeafHandlers

## [0.11.14] - 2025-06-09

### Added

- #v1 Add metadata async write / read methods.
- #v1 Add `start_async_mkdir` that begins async directory creation and returns
a `PathAwaitingCreation`.
- #v1 Add ArrayLeafHandler.
- #v1 Add CompatibleTypeHandler to allow v1 ArrayLeafHandlers working with V0
  TypeHandlerRegistry.
- #v1 Add tests to cover v0 CheckpointManager->v1 Checkpointer migration use
 case.
- #v1 Add NumpyLeafHandler.
- #v1 Add NumpyLeafHandler into CompatibleTypeHandler.
- #v1 Add ScalarLeafHandler and it into CompatibleTypeHandler.

### Fixed

- Fix JAX random key restoration with StandardRestore.

### Changed

- #v1 Allow `load_pytree` to specify the checkpointable_name, with `pytree` as
 default value.
- #v1 Use an implementation of `load` independent of `v0.Checkpointer`.
- #v1 Use an implementation of `save` independent of `v0.Checkpointer`.
- #v1 Distinguish between `force` and `overwrite` save params.
- Enable `pinned_host_transfer` for GPU backend only.
- #perf Add/update logs with time logs for save paths.
- Simplify str of NameFormats.
- `LoadCheckpointFromOrbax()` expects correct `weight_names` and `weight_specs`.

## [0.11.13] - 2025-04-30

### Changed

- Support for `jax.sharding.PositionalSharding` has been removed since it has
  been deprecated in JAX and will be deleted in a future JAX release. Please
  use `jax.NamedSharding` instead.
- #v1 Reverse handler resolution order so that the most recently registered
handler is preferred when multiple handlers would do the job. This increases
convenience for users - if they are registering their own handlers then they
would want them to be preferred.

### Added

- #v1 Add `JsonHandler`.
- #v1 Add `training.Checkpointer`.
- #v1 Add checkpointables support for `training.Checkpointer`.
- `PartsOf` structure which holds a PyTree whose leaf nodes may be missing.
- #v1 Add compatibility tests for save-by-v0-load-by-v1 and also fix code.
- #v1 Enable v1 users to load v0 checkpoints without using contexts.
- #v1 Add `StatefulCheckpointable` and `GenericHandler`.

## [0.11.12] - 2025-04-09

### Added

- #v1 Add `checkpointables_metadata` function and simplify PyTree
metadata presentation.
- #v1 Add v1 to `orbax/checkpoint/__init__.py`.

## [0.11.11] - 2025-04-07

### Changed

- Asynchronous directory creation enabled by default.
- Remove `CheckpointInfo.is_locked` as it is not used anymore.

### Added

- #v1 Add metadata free functions.
- #v1 Add CheckpointableHandlerRegistry and handler resolution functions.
- #v1 Add `CompositeHandler`, similar to the current
`CompositeCheckpointHandler`, but different in the sense that it will not be
visible to the user and is solely a helper layer for save/load functions.
- #v1 Implement `save/load_checkpointables` in terms of V0 APIs. Resolve
handlers for checkpointables using V1 APIs.
- `ocp.PLACEHOLDER` for partial restoration.
- `tree_trim` utility function for matching PyTree structures.
- #v1 Add `ProtoHandler`.

### Changed

- Improve `NameFormat.find_all` performance using
 `single_host_load_and_broadcast` strategy.

## [0.11.10] - 2025-03-20

### Added

- Add `fallback_sharding` option to `StandardRestoreArgs` to support restoring
on different topologies easily.
- `Placeholder` value that can be used during `PyTreeCheckpointHandler` restoration to partially restore a PyTree.

### Fixed

- Fix pinned_host loading by ensuring `memory_kind` is carried through when creating in-memory buffers.

## [0.11.9] - 2025-03-17

### Added

- `ocp.tree.utils.tree_difference()` to compute differences between PyTrees.

### Changed

- Re-enable `replica_parallel` feature. This was previously disabled due to
performance concerns, but these have been identified as relating to the
`pinned_host_transfer` feature instead.
- #v1 Refactor `Context` and expand global options.
- #v1 Add `CheckpointMetadata` definitions.
- Enable `ArrayMetadata` persistence globally.
- #emergency-checkpoint Create root directory automatically.

## [0.11.8] - 2025-03-07

### Changed

- Set JAX version requirement back to >= 0.5.0.

## [0.11.7] - 2025-03-06

### Added

- V1: `PyTreeHandler`.
- V1: Introduce Context and its usage as a contextmanager.
- V1: Add `ArrayStorageOptions` to customize per-leaf saving behavior for
arrays (e.g. `dtype`).
- `PlaceholderHandler` type handler for partial restoration project. Placeholder
values will not be loaded during checkpoint restoration.

### Fixed

- Fix RESOURCE_EXHAUSTED while writing array_metadatas.
- Internal/external mirroring was temporarily broken and a few changes were not
pushed when intended (since 2-27). However, there are only a few such changes,
which are reflected in the CHANGELOG.
- Fix ReadTheDocs breakage and fix a number of formatting issues.
Refactor `orbax.checkpoint.logging` in service of this objective.

### Changed

- Improve `Cannot serialize host local jax.Array` error message.

### Added

- support saving and restoring jax.random.key() in PyTree.
- `CheckpointableHandler` for V1.
- Support single-slice checkpointing in `emergency.CheckpointManager`.

## [0.11.6] - 2025-02-20

### Added

- Allow concurrent CheckpointManager.save requests and wait_until_finished.
- `orbax.checkpoint.experimental.v1` code structure, particularly including core
types and save/load free functions.
- Introduce `AsyncResponse` in V1 library. This is not much different from
`ocp.Future`, but serves as a generic type representing a result from a
background task. It also incorporates typing, in contrast with its predecessor.

### Changed

- Move process_metadata to the checkpoint_num directory
- Updated default JAX version requirement to 0.5.0.

## [0.11.5] - 2025-02-10

### Fixed

- Enable OCDBT read coalescing for remote storage

### Added

- `ocp.metadata.get_step_metadata(path)` to public api.

## [0.11.4] - 2025-02-07

### Changed

- Updated orbax-checkpoint PyPI package to exclude tests.

### Fixed

- `AsyncCheckpointer` completion logging, to log on all hosts instead of just
the leader.

## [0.11.3] - 2025-02-06

### Changed

- Improve local file read by disabling OCDBT read coalescing

## [0.11.2] - 2025-02-04

### Added

- `custom_metadata` parameter to `Checkpointer.save()`,
`AsyncCheckpointer.save()`, and `CheckpointManager.save()`, which saves a custom
dict of user metadata to `StepMetadata`.

### Fixed

- Add a check to prevent zero-sized arrays from being saved. This behavior
already resulted in an error, but it was one that was difficult to parse.

## [0.11.1] - 2025-01-28

### Changed

- Add user-facing `TreeMetadata` object returned by
`PyTreeCheckpointHandler.metadata`. This object mimics an ordinary PyTree to
make the change unnoticeable to most users, but also has additional accessible
properties not included in any tree mapping operations.
- `Checkpointer.save()`, `AsyncCheckpointer.save()` also saves `StepMetadata`.
- Added Github actions CI testing using Python versions 3.10-3.13.
- Standardize naming of the "custom metadata" field (user-supplied metadata) as
`custom_metadata`.
- Add `SaveDecisionPolicy` to better encapsulate various options around choosing
whether or not to perform a save at a particular step.

### Added

- The ability to specify a custom `snapshot_dir` in `checkpoints_iterator`.
- `CommitFutureAwaitDirectorySignals`, `CommitFuture` and
`HandlerAwaitableSignal` for signalling between Checkpointing layers to enable
async directory creation.
- Added `AsyncOptions.create_directories_asynchronously` flag to enable creating
directories asynchronously.
- A policy that allows for checkpointing as often as possible, as long as a
save is not already in progress (continuous checkpointing).
- User-provided custom PyTree metadata.

### Fixed

- Fix a bug where snapshots are not released by `wait_for_new_checkpoint` when a
custom `snapshot_dir` is specified.
- Disable `raise_array_data_missing_error` in `PyTreeCheckpointHandler`
restoration code path.
- Partially revert `replica_parallel` feature, as performance regressions have
been observed. Note also that simply setting `use_replica_parallel=False` does
not fix the issue.
- Disable `enable_pinned_host_transfer` option as poor performance has been
observed.

## [0.11.0] - 2024-12-30

### Fixed

- Resolve default value of Handler.typestr if method missing.

### Added

- Add announcement for grain version compatibility. See
 https://github.com/google/orbax/issues/1456.


## [0.10.3] - 2024-12-19

### Added

- `HandlerTypeRegistry` that globally tracks all handler types, which will allow
`CompositeCheckpointHandler.metadata()` to retrieve item metadata by
default-constructing `CheckpointHandler`s when they're listed in the saved
`StepMetadata` but aren't found in the checkpoint.

### Fixed

- Ignore not-exists and not-dir errors while building step metadata in
 _StandardNameFormat.
- Fix `Self` import in `replicator_checkpoint_manager.py`.

### Changed

- Return `StepMetadata` from `CompositeCheckpointHandler.metadata()`.

## [0.10.2] - 2024-12-04

### Added
- Tests and documentation for `abstract_arrays`.
- [Experimental Feature] Support `empty NamedTuple` leaf when
 `PyTreeMetadataOptions.support_rich_types=true`.
 - Enable `replica_parallel` saving. Uses cache to avoid recomputing
`devices_indices_map` for arrays with identical shape/sharding.
- Add Layout support to args.StandardRestore
- [emergency checkpoint] Open-source emergency checkpointing integration test
script.

### Fixed
- Fix namedtuple empty value typestr when experimental support_rich_types is
 disabled again after enabling it.
- Fix metadata ser/deser of jax registered container nodes like flax.struct.
- [emergency checkpoint] Fix local restore by re-mapping device ids directly
instead of inferring them from how process indexes changed across restarts
with some false assumptions.
- Avoid duplicate reads of the same index when loading.
- Fixed a regression bug on restoration with custom Layout.

### Changed
- Coordination service now supports barrier reuse - eliminate some barrier name
complexity, including counters.


## [0.10.1] - 2024-11-22

### Added
- [emergency checkpoint] Add ReplicatorCheckpointManager implementation for
interoperating with replicator service provided by GKE (or theoretically,
any other similar service).

### Fixed
- Correct issue where emergency checkpoint debug logging was not OSS-ed.

## [0.10.0] - 2024-11-21

### Added
- Add `RootMetadata` and `StepMetadata` classes as ways for the user to
interface with checkpoint metadata at various levels.
- Add `root_metadata_serialization`, and `step_metadata_serialization` modules
that contain utilities to perform de/serialization for `RootMetadata` and
`StepMetadata`.
- `ReplicaSlice`/`ReplicaSlices` construct to facilitate saving replicated
arrays.
- Added restoring with custom jax.experimental.layout.Layout support
- Add experimental `PyTreeMetadataOptions` to manage rich types in pytree
 checkpoint metadata.
- Create a separate namespace package `orbax.checkpoint.testing` for exporting
 test objects.
- `replica_parallel` saving that allows arrays with replicated shards to be
saved cooperatively by multiple hosts.
- [Experimental Feature] Support `NamedTuple` and `Tuple` nodes in PyTree
 metadata.
- Add validation to prevent loading an array index that was never written to.
- Add logging to detect missing chunks to emergency checkpointing to facilitate
local checkpoint debugging.

### Changed
- Refactor metadata/tree_test.py and move common test types to
 `test_tree_utils.py` for better reusability.
- [emergency checkpoint] Break out mesh construction and process ID metadata
utils into a separate file.
- Rename `CheckpointMetadataStore` to `MetadataStore`, and change methods to
accept and return metadata as dictionaries.
- Move `Checkpointer` implementations to `_src`.
- Add/Update tests for `is_empty_or_leaf` and `is_empty_node`.
- Refactor and restructure constructs from `type_handlers.py` and `metadata`
 package to avoid circular dependencies.

## [0.9.1] - 2024-11-11

### Added
- Introduce `CheckpointManagerOptions.should_keep_fn` as an alternative to
 `CheckpointManagerOptions.keep_period`.

### Fixed
- Fix readthedoc build failures on source files in `_src`

## [0.9.0] - 2024-11-08

### Changed
- Create `Composite` class, which `CompositeArgs` now subclasses.
- Move `tree` to `_src`.
- Move `serialization` to `_src`.
- Move `type_handlers` to `_src/serialization`
- Add notes to Barrier error `XlaRuntimeError(DEADLINE_EXCEEDED)` with
 actionable info.
- Make `NameFormat.find_all` impls concurrent.
- Move `path` package under `_src` package.
- Updated readthedoc

### Fixed
- Emergency checkpoint: use JAX for global_max and combine multiple broadcasts
into one for `saved` bool broadcast. This should alleviate concerns about
broadcasting using the distributed system at large scale.
- Emergency checkpoint: compile broadcast function once at init.

### Fixed
- Fix emergency checkpointing issue arising when repeatedly restoring from a
local checkpoint. Process ID remapping may happen in different ways on
subsequent times, so we need to ensure new process metadata is saved with every
checkpoint, so we can recover the process ID mapping used to save that
checkpoint.

## [0.8.0] - 2024-10-29

### Fixed
- bytes_per_sec calculation needs different values for read and write.

### Added
- Add a `SaveArgs` option that allows disabling pinned host transfer on a
per-array basis. UPDATE: Modify `enable_pinned_host_transfer` option to be
provided once for the entire pytree, since it's not really reasonable to
customize this on a per-array level.

### Changed
- Rename `CheckpointManager._single_item` to `CheckpointManager._default_item`.
- Add `strict` option in `ArrayRestoreArgs`, defaulting to True. This prevents
arrays from being accidentally padded or truncated when restoring.
- Move `multihost` implementations to `_src`. Commonly used symbols are still
exported in the same way.
- Use `Fragments` for serialization.
- Set `AsyncOptions.timeout_secs` default value to 10 minutes.
- De-duplicate `get_ts_context` usages and move to ts_utils.
- Move `logging` to `_src`.
- Move `metadata` to `_src`.

## [0.7.0] - 2024-10-07

### Removed
- Support for Python 3.9.

### Changed
- Modernize type annotations using `from __future__ import annotations`.
- Adjust `CompositeCheckpointHandler` behavior for unregistered items and empty
  `args`. Now, `metadata` only returns entries for which an item actually
  exists in the checkpoint. `restore` will raise an error if a requested item
  does not exist, and will attempt to restore all existing items if empty `args`
  are provided.
- Allow registering an item in `DefaultCheckpointHandlerRegistry` without
  providing the actual handler, as long as the provided args correspond to a
  globally registered handler. This allows for slightly reduced verbosity if we
  just want to ensure an association between an item name and args/handler.
- Refactor to extract a separate module, `asyncio_utils`, for asyncio helper
 functions from `path/async_utils` module.
- Rename `CheckpointMetadata` to `StepMetadata`.

### Added
- Support nested `asyncio.run` with `nest_asyncio` library.
- Support empty tuple values in checkpointable params.


## [0.6.4] - 2024-09-17

### Fixed
- Return empty dict (instead of None) as metadata for *empty* Mapping typed
 unregistered types.
- Restore to user mesh after emergency checkpoint local restoration.

 ### Changed
- Removed `write_chunk_shape` and `read_chunk_shape` from `SaveArgs`.
- Rename `is_supported_empty_aggregation_type` and
  `is_supported_aggregation_type` functions.
- Adjust `CompositeCheckpointHandler` behavior for unregistered items and empty
  `args`. Now, `metadata` only returns entries for which an item actually
  exists in the checkpoint. `restore` will raise an error if a requested item
  does not exist, and will attempt to restore all existing items if empty `args`
  are provided.
- Use `CheckpointHandlerRegistry` when `CheckpointManager` is
  default-constructed, e.g. `CheckpointManager(directory)`. Previously this
  usage pattern would default to single-item mode, but the new behavior will
  default to lazy initialization. The change should be a no-op for existing
  users.


## [0.6.3] - 2024-09-10

## [0.6.2] - 2024-09-10

### Added
- Support for `replica_axis_index = 0,1` in `broadcast_one_replica_to_all`.

### Fixed
- Only allow main-thread to reset Save Finalize error.
- Allow Orbax save to run from a non-main thread but continue to support
 concurrent access to wait_until_finished().

### Changed
- Refactor `merge_ocdbt_per_process_files()` to return explicitly without
 merging if none of the child subdir names match `ocdbt.process_` pattern.
- Moved `_choose_chunk_shape` from `type_handlers.py` to a new internal module.

### Added
- Validate merged and saved checkpoint params by comparing it against its
 `.zarray` counterpart.
- Added private utilities to work with array fragments (`_src/arrays`).
- Log `orbax-checkpoint version` when Checkpointer instances are created.
- NamedShardingMetadata includes full device mesh which will be used for
  restoration when sharding in restore_args is not provided.

## [0.6.1] - 2024-08-22

### Added
- Common public types to annotate pytrees.
- Add utils that act as a heuristic for detecting standard Orbax checkpoints.
- Add memory-based rate limiting support during save.
- Metrics to track bytes/sec during save and restore.
- Added `handler_registration.HandlerRegistry` to store custom mappings between
  checkpointable items and checkpoint args to a checkpoint handler.
- Added `handler_registration.HandlerRegistry` argument to
  `CompositeCheckpointhandler`. This replaces the `item_names` and
  `items_and_handlers` arguments for configuring items and handlers.
- Added `handler_registration.HandlerRegistry` argument to
  `CheckpointManager`. This replaces the `checkpointers`, `item_names` and
  `items_and_handlers` arguments for configuring items and handlers.
- Raise error if `AsyncOptions.post_finalization_callback` is given but
 `CheckpointManager._checkpinter` is not async.
- Add Tensorstore memory debug logs when debug message is turned on, eg.
  `--vmodule=type_handlers=1`.


## [0.6.0] - 2024-08-19

### Fixed
- Fix callsites of handler.async_save to handle returned None.

### Changed
- Improve logging by adding jax_process, error logs in threads and more...
- Improvements to blocking save time, as a result of moving file open operations
into the background.
- Consolidate usages of `MultiprocessingOptions` and `AsyncOptions`.
- Formalize `StandardCheckpointer` as an `AsyncCheckpointer` that doesn't
require a `CheckpointArgs` object, and instead allows directly passing the
state and extra args (breaking change).
- Add log messages to improve debugging with jax process and threads.
- Improve logging to turn on DEBUG log per source file.
- Update bytes/sec metrics to be per-host.
- Parallelize many directory creations using asyncio. Also reduce the number
of `asyncio.run` calls by moving `async` functions higher in the stack.
- Allow CheckpointManager.wait_until_finished to be called concurrently.

### Added
- Add utils that act as a heuristic for detecting standard Orbax checkpoints.
- Add memory-based rate limiting support during save.
- Metrics to track bytes/sec during save and restore.

### Removed
- Preliminary concurrent creation of parameter directories from
`PyTreeCheckpointHandler` (in non-OCDBT case).


## [0.5.23] - 2024-07-26

### Changed
- In `CheckpointManager.restore`, allow specifying `step=None`, which
automatically restores from latest.
- Emergency checkpointing bug-fixes for CPU backend.

## [0.5.22] - 2024-07-19

### Changed
- Move `get_param_names` to tree utils.
- Move all work for `_write_metadata_file` into a background thread to avoid
O(n) computation in building metadata.

### Fixed
- Adjust user metadata construction so that older checkpoints with some values
in the msgpack file can still return metadata (though the metadata for these
entries will not return information about array properties).

## [0.5.21] - 2024-07-12

### Added
- Rolled forward change to improve TensorStore I/O efficiency.
- Memory efficient broadcasting from one model replica to others.

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
- Add memory-based rate limiting support during save.

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
