# Orbax

[Orbax](https://orbax.readthedocs.io/en/latest/) is a
namespace providing common utility libraries for JAX users.

## [Documentation](https://orbax.readthedocs.io/en/latest/)

## Checkpointing

`pip install orbax-checkpoint` (latest PyPi release) OR

`pip install 'git+https://github.com/google/orbax/#subdirectory=checkpoint'` (from this repository, at HEAD)

`import orbax.checkpoint`

Orbax includes a checkpointing library oriented towards JAX users, supporting a
variety of different features required by different frameworks, including
asynchronous checkpointing, various types, and various storage formats.
We aim to provide a highly customizable and composable API which maximizes
flexibility for diverse use cases.


## Exporting

`pip install orbax-export` (latest PyPi release) OR

`pip install 'git+https://github.com/google/orbax/#subdirectory=export'` (from this repository, at HEAD)

`import orbax.export`

Orbax also includes a serialization library for JAX users, enabling the exporting of JAX models to the TensorFlow SavedModel format.

Note that `orbax-export` requires TensorFlow, but does not include it by default to allow for flexibility in version choice. If you wish to install with standard TensorFlow, please use `pip install orbax-export[all]`.


## Support

Contact orbax-dev@google.com for help or with any questions about Orbax!

### History

Orbax was initially published as a catch-all package itself. In order to minimize dependency bloat for users, we have frozen that package at `orbax-0.1.6`, and will continue to release future changes under the domain-specific utilities detailed above (e.g. `orbax-checkpoint`). 

As we have preserved the orbax namespace, existing import statements can remain unchanged (e.g. `from orbax import checkpoint`).