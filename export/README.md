# Orbax Export

`pip install orbax-export` (latest PyPi release) OR

`pip install 'git+https://github.com/google/orbax/#subdirectory=export'` (from this repository, at HEAD)

`import orbax.export`

Note that Orbax depends on TensorFlow, which is not included in the above commands.
You should install TensorFlow separately, e.g. `pip install tensorflow` or `pip install tensorflow-cpu`.

Orbax includes a serialization library for JAX users, enabling the exporting of
JAX models to the TensorFlow SavedModel format. 

To get started, check out our [documentation](https://github.com/google/orbax/blob/main/docs/export.md).
