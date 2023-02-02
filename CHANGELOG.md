# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add "standard user recipe" to documentation.
- Add unit tests using mock to simulate preemption.
- Logging to increase transparency around why checkpoints are kept vs. deleted.

### Changed
- Remove CheckpointManager update API.

### Fixed
- Preemption when using keep_time_interval caused the most recent steps before
preemption to be kept, despite not falling on the keep time interval.

## [0.1.0] - 2022-01-30

### Added
- A util function that constructs restore_args from a target PyTree.
- CheckpointManager `delete` API, which allows deleting an existing step.
- Made dev dependencies optional to minimize import overhead.

### Changed
- Refactored higher-level utils in checkpoint_utils, which provides user-convenience functions.
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

