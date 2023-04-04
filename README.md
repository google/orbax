# Orbax

[Orbax](https://github.com/google/orbax/blob/main/docs/index.md) is a
namespace providing common utility libraries for JAX users.

## Checkpointing

`pip install orbax-checkpoint`

`import orbax.checkpoint`

Orbax includes a checkpointing library oriented towards JAX users, supporting a
variety of different features required by different frameworks, including
asynchronous checkpointing, various types, and various storage formats.
We aim to provide a highly customizable and composable API which maximizes
flexibility for diverse use cases.

To get started, check out our [documentation](https://github.com/google/orbax/blob/main/docs/checkpoint.md).

Check out our [colab](http://colab.research.google.com/github/google/orbax/blob/main/checkpoint/orbax//checkpoint/orbax_checkpoint.ipynb) for a hands-on introduction.

## Exporting

`pip install orbax-export`

`import orbax.export`

Orbax also includes a serialization library for JAX users, enabling the exporting of JAX models to the TensorFlow SavedModel format. 

To get started, check out our [documentation](https://github.com/google/orbax/blob/main/docs/export.md).
<!-- TODO(dinghua): Add information on export library. -->

## Support

Contact orbax-dev@google.com for help or with any questions about Orbax!

### History

Orbax was initially published as a catch-all package itself. In order to minimize dependency bloat for users, we have frozen that package at `orbax-0.1.6`, and will continue to release future changes under the domain-specific utilities detailed above (e.g. `orbax-checkpoint`). 

As we have preserved the orbax namespace, existing import statements can remain unchanged (e.g. `from orbax import checkpoint`).