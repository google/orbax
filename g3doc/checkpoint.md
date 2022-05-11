# Checkpointing

go/orbax/checkpoint

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'cpgaffney' reviewed: '2022-05-10' }
*-->

[TOC]

## Introduction

Orbax provides a flexible and customizable for managing checkpoints for various
different objects.

### CheckpointManager

[`CheckpointManager`]() is the highest-level object provided by Orbax for
checkpointing, and is generally the interface that should be most often used to
interact with checkpoints.

This manager allows saving and restoring any object for which a `Checkpointer`
implementation exists (see [below](#checkpointer)). This may include objects
such as [JAX PyTree](https://jax.readthedocs.io/en/latest/pytrees.html),
[`tf.data.Iterator`](https://www.tensorflow.org/api_docs/python/tf/data/Iterator),
or JSON-format dictionaries.

Here is a simple usage example:

```py
mngr = CheckpointManager('path/to/directory/', PyTreeCheckpointer())
item = {'a': 0, 'b': {'c': 1, 'd': 2}}
for step in range(10):
  mngr.save(step, item)
restored = mngr.restore(step)
```

This allows saving and restoring a single PyTree object using a
`PyTreeCheckpointer`. A more complex case allows managing several different
objects and customizing `CheckpointManager` behavior.

```py
def best_fn(metrics: Mapping[str, float]) -> float:
  return metrics['accuracy']

options = CheckpointManagerOptions(max_to_keep=5, best_fn=best_fn, mode='max')
checkpointers = {'state': PyTreeCheckpointer(), 'dataset': DatasetCheckpointer()}
mngr = CheckpointManager('path/to/directory/', checkpointers, options=options)

state = {'a': 0, 'b': {'c': 1, 'd': 2}}
dataset = iter(tf.data.Dataset(range(2)))
for step in range(10):
  mngr.save(step, {'state': state, 'dataset': dataset},
              metrics={'accuracy': ...})
restored = mngr.restore(step, items={'state': None, 'dataset': dataset})
restored_state, restored_dataset = restored['state'], restored['dataset']
```

In this example, we begin by specifying options for `CheckpointManager`, which
instruct it to keep a maximum of 5 checkpoints, and also to track metrics
associated with each checkpoint.

We can then give a dictionary of checkpointers with unique keys for every item
we want to save. Each key has a `Checkpointer` object as a value, which
instructs the `CheckpointManager` how to save the given object. When calling
save, we provide a dictionary with the same keys, each corresponding to an item
to be saved.

In this multi-object setting, we must also provide a dictionary of items to
restore. The value may be given as `None` if it is not needed by the underlying
`Checkpointer` to perform the restore operation. It then returns a dictionary of
restored objects with the same keys as provided.

Each `Checkpointer` may accept additional optional arguments. This can be passed
through from `CheckpointManager` to `Checkpointer` via `save_kwargs` and
`restore_kwargs`. For example:

```py
empty_state = jax.tree_map(lambda _: object(), pytree_state)
save_args = jax.tree_map(lambda _: SaveArgs(...), pytree_state)
restore_args = jax.tree_map(labmda _: RestoreArgs(...), pytree_state)

mngr.save(step, items={'state': pytree_state, ...},
            save_kwargs={'state': {'save_args': save_args}})
mngr.restore(step, items={'state': empty_state, ...},
            save_kwargs={'state': {'restore_args': restore_args}})
```

Both `save_kwargs` and `restore_kwargs` are nested dictionaries where the
top-level keys correspond to the items to be checkpointed. The values are
dictionaries of optional arguments that are provided to the `Checkpointer` as
keyword arguments.

Other APIs include:

*   `all_steps`: returns an unsorted list of integers of steps saved in the
    `CheckpointManager`'s directory.
*   `latest_step`: returns the most recent step.
*   `best_step`: returns the best step as defined by the `best_fn` which runs
    over provided metrics. Returns the latest step if `best_fn` is not defined.
*   `should_save`: returns whether save should be performed or skipped at the
    given step. This dependson factors such as the most recent saved step as
    well as

### Checkpointer

[`Checkpointer`]() provides an interface which can be implemented to provide
support for saving and restoring a particular object. Several objects are
supported by default in Orbax (see [below](#checkpointer-implementations)).

The class provides `save`/`async_save` and `restore`/`async_restore` APIs which
save or restore an `item` either synchronously or asynchronously to a provided
`directory`.

Ideally, the asynchronous versions of the methods should be implemented, as this
allows `CheckpointManager` to schedule operations in parallel. However,
`CheckpointManager` can use the synchronous methods if no asynchronous version
is provided. By default the sync version simply calls the async version before
running a global sync.

### Checkpointer Implementations

#### PyTreeCheckpointer

[`PyTreeCheckpointer`]() allows checkpointing PyTrees consisting of scalars,
np/jnp arrays, or
[`GlobalDeviceArray`]()
(`GDA`). Note that this class provides support for device-partitioned arrays via
`GDA`. Other values are expected to be replicated across devices.

For saving and restoring, `PyTreeCheckpointer` provides optional arguments on a
per-element basis via `SaveArgs` and `RestoreArgs`. This means that parameters
are provided on an individual basis for each element in the PyTree.

`SaveArgs` parameters include:

*   `use_flax`: if true, saves the given parameter using flax.serialization to a
    unified checkpoint file. Must be false if the given array value is a GDA.
*   `dtype`: if provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).

`RestoreArgs` parameters include:

*   `as_gda`: if true, restores the given paramater as a GlobalDeviceArray
    regardless of how it was saved. If the array was not saved as a GDA, mesh
    and mesh_axes are required.
*   `mesh`: the device mesh that the array should be restored as. If None, uses
    a linear mesh of jax.devices.
*   `mesh_axes`: the mesh_axes that the array should be restored as. If None,
    fully replicates the array to every device.
*   `global_shapes`: the global shape that the array should be restored into. If
    not provided, the shape will be restored as written.
*   `lazy`: if True, restores using [LazyArray](#lazyarray). The actual read
    operation will not be performed until `get` is called for the restored
    LazyArray
*   `dtype`: if provided, casts the parameter to the given dtype after
    restoring. Note that the parameter must be compatible with the given type
    (e.g. jnp.bfloat16 is not compatible with np.ndarray).

`PyTreeCheckpointer` will create an individual directory for each nested key.
The exact naming of per-parameter directories can be customized by overriding
`_get_param_infos`. These parameters are saved using
[Tensorstore](https://google.github.io/tensorstore/). There is an additional
`checkpoint` file that will be created using
[`flax.serialization`](https://flax.readthedocs.io/en/latest/flax.serialization.html).
This stores the PyTree structure, as well as any parameters for which `use_flax`
was True at save time. Individual directories will not be created for these
parameters.

For `restore`, `item` is an optional argument because the PyTree structure can
be recovered from the saved checkpoint if `item` is a dictionary. However, if
`item` is an object other than `dict`, `item` should be provided in order to
restore the object structure.

#### JsonCheckpointer

[`JsonCheckpointer`]() is provided as a way to checkpoint nested dictionaries
that can be serialized in JSON format. This can be useful as a way to store
checkpoint metadata. For example, `CheckpointManager` uses this class to store
the metrics used to evaluate relative checkpoint quality.

Note that presently, this class does not implement async APIs.

#### DatasetCheckpointer

[`DatasetCheckpointer`]() is designed for saving and restoring
`tf.data.Iterator`. It does this using
[`tf.train.Checkpoint`](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint).

Unlike the preceding classes, this class always requires `item` to be provided
when restoring (in the other classes it is optional or unused).

Note that presently, this class does not implement async APIs, since it relies
on `tf.train.Checkpoint`.

### Utilities

#### LazyArray

[`LazyArray`]() provides a mechanism for delayed loading of arrays from a
checkpoint. If a parameter is restored as a `LazyArray` (in
`PyTreeCheckpointer`, setting `RestoreArgs.lazy = True`), the restored object
will not yet have done the work of actually loading the parameter from
Tensorstore.

The actual loading will only be done when `.get()` is called on the `LazyArray`.

Of course, for parameters saved using `flax.serialization` into a file
containing many parameters, the loading does happen eagerly, regardless of
whether `LazyArray` is used. However, parameters saved in this way should
typically be small.

#### Transformations

The [`transform_utils`]() library provides functions to allow structural PyTree
transformations, which can facilitate migrations between different checkpoint
versions.

The API consists of a `Transform` class and an `apply_transformations` function.
`Transform` consists of the following elements:

*   `origin_name`: Denotes the original name of the key. Represented using a
    tuple, where each successive element represents a nested key in a nested
    dictionary. May also provide a string, which will be interpreted as a tuple
    of length 1 (no nesting). Note: not needed if value_fn is provided.
*   `in_checkpoint`: Indicates whether a parameter is expected to be present in
    the saved checkpoint. Will raise an error if the parameter was expected, but
    is not present.
*   `value_fn`: A function accepting a PyTree and returning any value. The
    PyTree argument will be the original PyTree, and the function should return
    the value of the key in the new PyTree.

This can be represented with some simple examples:

`{'a': Transform(origin_name='b')}`

This denotes that the original key was named 'b', but we are changing it to 'a'.

`{'a': Transform(value_fn=lambda kv: kv['b'] * 2)}`

This signifies that the new key 'a' is the old key 'b' multiplied by two.

The `apply_transformations` function accepts an original PyTree and a PyTree of
`Transform` objects. The function will return a PyTree matching the
`transformations` tree.

For example:

```py
original_tree = {
  'a': 1,
  'b': {
    'c': 5,
    'd': [0, 1, 2, 3]
  },
}
transforms = {
  'a1': Transform(origin_name='a'),  # rename
  'a1': Transform(value_fn=lambda kv: kv['a']),  # another way of doing above
  'b': {
    'c': Transform(value_fn=lambda kv: kv['b']['c'] * 2)  # doubled original
    # drop b/d
  },
   # Copy original into multiple new keys
  'c1': Transform(origin_name=('b', 'c')),
  'c2': Transform(origin_name=('b', 'c')),
  # one to many mapping
  'x': Transform(value_fn=lambda kv: kv['b']['d'][0]),
  'y': Transform(value_fn=lambda kv: kv['b']['d'][1:]),
  # many to one mapping
  'z': Transform(value_fn=lambda kv: kv['a'] * 2 + sum(kv['b']['d'])),
  # create a new key not in original
  'new': Transform(in_checkpoint=False),
}

transformed_tree = apply_transformations(original_tree, transforms)
```
