# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `orbax_export.config_class` API to set global configs.
- Add `orbax_export.typing` API
- Require jax version >=0.4.30 as the jax.export API was introduced in this release.


## [0.0.4] - 2024-1-19

### Added
- Add a `update_variables` method to JaxModule class to allow variable update.
- Add an example of periodically exporting JAX model during training.

### Changed
- Make `ValidationManager` and `ExportManager` share same API interface.
- Use different `tf.Graph`s in `utils.from_saved_model` when loading a
SavedModel and reconstructing its signature as python callable.
- Migrate `tf.types.experimental.GenericFunction` usages to
`tf.types.experimental.PolymorphicFunction`, as TensorFlow 2.15.0 renamed the
class. As a result, this will require TensorFlow >= 2.15.0.
- Use CPU device to convert params to tf.Variable, since TensorFlow by default
uses GPUs when available to create the variable copies. This can lead to GPU OOM
when using Orbax Exporter during training.

## [0.0.3] - 2023-08-01

### Added
- The `TensorSpecWithDefault` class and utility functions for adding default
values to model signatures.
- The `reset_context` argument for `initialize_dtensor` for resetting TF context.
- The `signature_overrides` argument for `ExportManager.save` for overriding the
signatures to be exported.
- Limited support for product-sharded (i.e., a tensor axis sharded across
multiple mesh axes) parameters.
- Escaping invalid character "~" from `JaxModule` variable names.


## [0.0.2] - 2023-04-04

### Added
- Released `orbax-export`, a serialization library for JAX users enabling the
exporting of JAX models to the TensorFlow SavedModel format.

