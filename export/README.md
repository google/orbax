# Orbax Export

`pip install orbax-export` (latest PyPi release) OR

`pip install 'git+https://github.com/google/orbax/#subdirectory=export'` (from this repository, at HEAD)

`import orbax.export`

Orbax includes a serialization library for JAX users, enabling the exporting of JAX models to the TensorFlow SavedModel format.

Note that `orbax-export` requires TensorFlow, but does not include it by default to allow for flexibility in version choice. If you wish to install with standard TensorFlow, please use `pip install orbax-export[all]`.

To get started, check out our [documentation](https://orbax.readthedocs.io/en/latest/orbax_export_101.html).
