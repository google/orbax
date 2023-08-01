# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

