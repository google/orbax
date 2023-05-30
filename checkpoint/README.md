# Orbax Checkpointing

`pip install orbax-checkpoint` (latest PyPi release) OR

`pip install 'git+https://github.com/google/orbax/#subdirectory=checkpoint'` (from this repository, at HEAD)

`import orbax.checkpoint`

Orbax includes a checkpointing library oriented towards JAX users, supporting a
variety of different features required by different frameworks, including
asynchronous checkpointing, various types, and various storage formats.
We aim to provide a highly customizable and composable API which maximizes
flexibility for diverse use cases.

To get started, check out our [documentation](https://github.com/google/orbax/blob/main/docs/checkpoint.md).

Check out our [colab](http://colab.research.google.com/github/google/orbax/blob/main/checkpoint/orbax//checkpoint/orbax_checkpoint.ipynb) for a hands-on introduction.



