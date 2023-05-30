# Checkpointing

https://github.com/google/orbax/blob/main/docs/checkpoint.md


## Introduction

Orbax provides a flexible and customizable API for managing checkpoints for
various objects.

Check out our
[colab](http://colab.research.google.com/github/google/orbax/blob/main/checkpoint/orbax//checkpoint/orbax_checkpoint.ipynb)
for some hands-on examples.


## Getting Started

### Quick and Simple

The following example shows how you can synchronously save and restore a
[PyTree](https://jax.readthedocs.io/en/latest/pytrees.html).

```py
checkpointer = orbax.checkpoint.PyTreeCheckpointer()
# 'path/to/directory' should already exist, but 'checkpoint_name' folder should
# not.
# The leaves of `my_tree` may be a number of different types.
# See `PyTreeCheckpointHandler` documentation.
checkpointer.save('path/to/directory/checkpoint_name/', my_tree)
# If you want to restore any of the leaves as sharded arrays, you'll need some
# extra arguments. See `PyTreeCheckpointHandler` documentation.
restored = checkpointer.restore('path/to/directory/checkpoint_name/')
```

For more detailed information, see the [`Checkpointer`](#checkpointer) and
[`PyTreeCheckpointHandler`](#pytreecheckpointhandler) sections.

### Managing Checkpoints

Sometimes, you may have multiple different objects that you want to checkpoint.
You may also wish to benefit from more high-level management logic to keep track
of your checkpoints while training progresses.

```py
# Keeps a maximum of 3 checkpoints, and only saves every other step.
# See `CheckpointManager` documentation for more options.
options = CheckpointManagerOptions(max_to_keep=3, keep_period=2)
mngr = CheckpointManager(
          'path/to/directory/',
          {'state': PyTreeCheckpointer(), 'extra_params': PyTreeCheckpointer()},
           options=options)

for step in range(11):  # [0, 1, ..., 10]
  mngr.save(step, {'state': train_state, 'extra_params': extra_params})
restored = mngr.restore(10)
restored_state, restored_extra_params = restored['state'], restored['extra_params']

mngr.all_steps()  # [6, 8, 10]
mngr.latest_step()  # 10
mngr.should_save(11)  # False
```

For more detailed information, see [`CheckpointManager`](#checkpointmanager)
documentation.

### A Standard Recipe

In most cases, users will wish to save and restore a PyTree representing a model
state over the course of many training steps. Many users will also wish to do
this is a multi-host, multi-device environment.

NOTE: See [below](#pytreecheckpointhandler) for information on how to use
RestoreArgs and ArrayRestoreArgs.

```py
# Create PyTree state with leaves as sharded jax.Array.
# Since we're going to restore the state from a checkpoint, the leaf values can
# be randomly or zero initialized. This state merely serves to enforce the
# structure of the state we are going to restore, along with the sharding of
# individual parameters.
train_state = {
  ...
}
options = CheckpointManagerOptions(max_to_keep=3, keep_period=2)
mngr = CheckpointManager(
          'path/to/directory/', PyTreeCheckpointer(),
          options=options)

if mngr.latest_step() is not None:  # existing checkpoint present
  # Use convenience function to construct args.
  shardings = jax.tree_map(lambda x: x.sharding, train_state)
  restore_args = checkpoint_utils.construct_restore_args(
                    train_state, shardings)
  # Directly construct args.
  restore_args = jax.tree_map(
    lambda x: ArrayRestoreArgs(
        # Restore as object. Could also be np.ndarray, int, or others.
        restore_type=jax.Array,
        # Cast the restored array to a specific dtype.
        dtype=np.float32,
        mesh=x.sharding.mesh,
        mesh_axes=x.sharding.spec,
        # Padding or truncation may occur. Ensure that the shape matches the
        # saved shape!
        global_shape=x.shape,
    ),
    train_state)
  # Note the use of plural 'items' and 'restore_kwargs'. This is because we may
  # be managing multiple items, as shown in the previous section. It is also
  # valid to just have one item, as shown here.
  restored = mngr.restore(mngr.latest_step(), 
                items=train_state, restore_kwargs={'restore_args': restore_args})

start_step = 0 if mngr.latest_step() is None else mngr.latest_step() + 1
for step in range(start_step, start_step + num_steps):
  train_state = do_training(train_state)
  mngr.save(step, train_state)
```

## CheckpointManager

[`CheckpointManager`](https://github.com/google/orbax/tree/main/orbax/checkpoint/checkpoint_manager.py)
is the highest-level object provided by Orbax for checkpointing, and is
generally the interface that should be most often used to interact with
checkpoints.

This manager allows saving and restoring any object for which a
`CheckpointHandler` implementation exists (see [below](#checkpointhandler)).
This may include objects such as
[JAX PyTree](https://jax.readthedocs.io/en/latest/pytrees.html),
[`tf.data.Iterator`](https://www.tensorflow.org/api_docs/python/tf/data/Iterator),
JSON-format dictionaries, and others.

A `CheckpointHandler` should be used in conjunction with a
[`Checkpointer`](#checkpointer) object. This allows customizing the logic used
to save the checkpoint atomically and synchronously (or asynchronously).

Here is a simple usage example:

```py
mngr = CheckpointManager('path/to/directory/', Checkpointer(PyTreeCheckpointHandler()))
item = {'a': 0, 'b': {'c': 1, 'd': 2}}
for step in range(10):
  mngr.save(step, item)
step = 10
restored = mngr.restore(step)
```

This allows saving and restoring a single PyTree object using a
`PyTreeCheckpointHandler`, wrapped with a `Checkpointer`, which performs the
save synchronously. A more complex case allows managing several different
objects and customizing `CheckpointManager` behavior.

```py
def best_fn(metrics: Mapping[str, float]) -> float:
  return metrics['accuracy']

options = CheckpointManagerOptions(max_to_keep=5, best_fn=best_fn, best_mode='max')
handlers = {'state': AsyncCheckpointer(PyTreeCheckpointHandler()), 'metadata': Checkpointer(JsonCheckpointHandler())}
mngr = CheckpointManager('path/to/directory/', handlers, options=options)

state = {'a': 0, 'b': {'c': 1, 'd': 2}}
metadata = {'learning_rate': 0.001, 'version': 1.1, 'exp_name': 'best_exp_123'}
for step in range(10):
  mngr.save(step, {'state': state, 'metadata': metadata},
              metrics={'accuracy': ...})
# do something else
mngr.wait_until_finished()  # wait for async save to complete.
restored = mngr.restore(step, items={'state': None, 'metadata': None})
restored_state, restored_metadata = restored['state'], restored['metadata']
```

In this example, we begin by specifying options for `CheckpointManager`, which
instruct it to keep a maximum of 5 checkpoints, and also to track metrics
associated with each checkpoint.

We can then give a dictionary of checkpointers with unique keys for every item
we want to save. Each key has a `Checkpointer` object as a value, which in turn
wraps a `CheckpointHandler` object. This instructs the `CheckpointManager` on
how to save the given object. When calling `save`, we provide a dictionary with
the same keys, each corresponding to an item to be saved.

Note that `AsyncCheckpointer` can be used in conjunction with supported
`CheckpointHandler` subclasses. This allows the save operation to proceed in a
background thread while waiting for completion.

After saving several checkpoints, the directory will look like this:

```
path/to/directory
  0/
    state/
      # Binary msgpack file: stores aggregated parameters and entire PyTree
      # structure.
      checkpoint
      layer0/  # directory for each key in the PyTree
        <files generated by Tensorstore>
      layer1/
        ...
      .../
    metadata/
      <json file>
  1/
    ...
  2/
    ...
  .../
```

In this multi-object setting, we must also provide a dictionary of items to
restore. The value may be given as `None` if it is not needed by the underlying
`CheckpointHandler` to perform the restore operation. It then returns a
dictionary of restored objects with the same keys as provided.

Each `CheckpointHandler` may accept additional optional arguments. These can be
passed through from `CheckpointManager` to `Checkpointer` to `CheckpointHandler`
via `save_kwargs` and `restore_kwargs`. For example:

```py
empty_state = jax.tree_map(lambda _: object(), pytree_state)
save_args = jax.tree_map(lambda _: SaveArgs(...), pytree_state)
restore_args = jax.tree_map(lambda _: RestoreArgs(...), pytree_state)

mngr.save(step, items={'state': pytree_state, ...},
  save_kwargs={'state': {'save_args': save_args}})
mngr.restore(step, items={'state': empty_state, ...},
  save_kwargs={'state': {'restore_args': restore_args}})
```

Both `save_kwargs` and `restore_kwargs` are nested dictionaries where the
top-level keys correspond to the items to be checkpointed. The values are
dictionaries of optional arguments that are provided to `Checkpointer`, and then
to `CheckpointHandler`, as keyword arguments.

Other APIs include:

*   `directory` (property): returns the directory where checkpoints are kept.
*   `all_steps`: returns an unsorted list of integers of steps saved in the
    `CheckpointManager`'s directory.
*   `latest_step`: returns the most recent step.
*   `best_step`: returns the best step as defined by the `best_fn` which runs
    over provided metrics. Returns the latest step if `best_fn` is not defined.
*   `should_save`: returns whether save should be performed or skipped at the
    given step. This depends on factors such as the most recent saved step as
    well as the specified save interval.
*   `wait_until_finished`: waits for any incomplete save operations to complete
    by calling the same method for any `AsyncCheckpointer`s. This will be a
    no-op if no `Checkpointer`s are async.
*   `structure`: returns a dictionary with the same items as the checkpointers
    originally passed to the manager. Delegates to the underlying `Checkpointer`
    and then to `CheckpointHandler`. For any `CheckpointHandler` which does not
    implement the method, that key will simply not be present in the returned
    dict.
*   `metadata`: returns the global checkpoint metadata, if present. This
    metadata must be provided at `CheckpointManager` initialization time.

Configurable `CheckpointManagerOptions` include:

*   `save_interval_steps`: the interval at which checkpoints should be saved.
    Ensures checkpoints will only be saved every n steps. Defaults to 1.
*   `max_to_keep`: if provided, specifies the maximum number of checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints.
*   `keep_time_interval`: When more than max_to_keep checkpoints are present, an
    older checkpoint that would ordinarily be deleted will be preserved if it
    has been at least `keep_time_interval` since the previous preserved
    checkpoint. The default setting of `None` does not preserve any checkpoints
    in this way. For example, this may be used to ensure checkpoints are
    retained at a frequency of approximately than one per hour.
*   `keep_period`: If set, will not delete any checkpoint where checkpoint_step
    % keep_period == 0.
*   `best_fn`: if set, maintains checkpoints based on the quality of given
    metrics rather than recency. The function should accept a PyTree of metrics,
    and return a scalar value that can be used to determine the quality score of
    the checkpoint. If `max_to_keep` is also set, then the retained checkpoints
    will be kept based on their quality, as measured by this function.
*   `best_mode`: one of ['max', 'min']. The best metric is determine on the
    basis of this value.
*   `keep_checkpoints_without_metrics`: If False, checkpoints with metrics
    present are eligible for cleanup. Otherwise, they will never be deleted.
*   `step_prefix`: if provided, step directories will take the form
    f'{step_prefix}_<step>'. Otherwise, they will simply be an integer <step>.

These lists are not necessarily exhaustive. See the
[code](https://github.com/google/orbax/tree/main/orbax/checkpoint/checkpoint_manager.py) for
full details.

## Checkpointer

[`Checkpointer`](https://github.com/google/orbax/tree/main/orbax/checkpoint/checkpointer.py)
serves as an intermediate layer between the high-level APIs of
`CheckpointManager` and the lower-level, per-type logic of `CheckpointHandler`.
It's purpose is to provide a no-frills way to atomically save an individual
object, while also retaining its independence as a separate layer in order to
better support customization. This is best illustrated by
[`AsyncCheckpointer`](#asynccheckpointer), which provides generalized logic for
saving objects in a background thread.

This class may be a good option if you only want to save or restore a single
object from a specific directory, and do not care about extra functionality that
tracks steps or best metrics, for example.

`Checkpointer` only provides `save`, `restore`, and `structure` APIs. Each of
these ultimately delegates to underlying `CheckpointHandler` provided at
construction. For `save`, however, the `Checkpointer` ensures that the operation
will be atomic.

### AsyncCheckpointer

[`AsyncCheckpointer`](https://github.com/google/orbax/tree/main/orbax/checkpoint/async_checkpointer.py)
is similar in almost every way to `Checkpointer`, but the save operation happens
in a background thread, while returning immediately to allow the main thread to
do something else. However, the operation is guaranteed to be eventually atomic.

Unlike `Checkpointer`, which can wrap any `CheckpointHandler`,
`AsyncCheckpointer` can only wrap `AsyncCheckpointHandler`, because it requires
the async save method that this subclass provides.

Users should call `wait_until_finished` to block until completion of outstanding
save operations.

## CheckpointHandler

IMPORTANT: `CheckpointHandler` is not intended to be used alone, but only in
conjunction with `Checkpointer` or `CheckpointManager`.

[`CheckpointHandler`](https://github.com/google/orbax/tree/main/orbax/checkpoint/checkpoint_handler.py)
provides an interface which can be implemented to provide support for saving and
restoring a particular object. Several objects are supported by default in Orbax
(see [below](#checkpointhandler-implementations)).

The class provides `save` and `restore` APIs which save or restore an `item`
synchronously given a specific `directory`. The save operation should not be
atomic, since this functionality is handled by `Checkpointer`.

### Checkpointer vs. CheckpointHandler

The need for a division of labor between `Checkpointer` and `CheckpointHandler`
is not always obvious, but we have found that the design increases modularity
and reduces code duplication.

This is most obvious when it comes to async checkpointing. The logic required to
manage a background thread is complex, and we wish to centralize it in a single
location rather than requiring every user with a new type to implement
themselves in their own `CheckpointHandler`. We also wish to provide a
synchronous `Checkpointer` in a separate implementation rather than requiring
all users to go through `AsyncCheckpointer`. This object can be much simpler to
use and understand. However, we need an additional layer represented by the
`CheckpointHandler` to implement type-specific logic, so that `Checkpointer` and
`AsyncCheckpointer` may share code.

Finally, atomicity is handled at the `Checkpointer` layer, again so that it need
not be re-implemented for every custom type. Furthermore, logic ensuring
atomicity may be implemented in different ways on different file systems,
therefore requiring a more modular design.

### AsyncCheckpointHandler

A special interface inheriting from `CheckpointHandler`,
[`AsyncCheckpointHandler`](https://github.com/google/orbax/tree/main/orbax/checkpoint/async_checkpoint_handler.py)
provides an additional async method called `async_save`, which has a similar
interface to `save`, but with significant differences.

Awaiting `async_save` should perform a copy of the object data from device to
host. The method should then return a list of futures which, when run, should
complete the saving of the object from host to storage location.

All subclasses of `AsyncCheckpointHandler` can easily implement their `save`
method by calling `async_save`.

## CheckpointHandler Implementations

### PyTreeCheckpointHandler

[`PyTreeCheckpointHandler`](https://github.com/google/orbax/tree/main/orbax/checkpoint/pytree_checkpoint_handler.py)
allows checkpointing PyTrees consisting of scalars, np/jnp arrays, or
`jax.Array`. Note that this class provides support for device-partitioned arrays
via `jax.Array`. Other values are expected to be replicated across devices.

This is a subclass of `AsyncCheckpointHandler`, which means that it allows
asynchronous saves via `async_save`.

For saving and restoring, `PyTreeCheckpointHandler` provides optional arguments
on a per-element basis via `SaveArgs` and `RestoreArgs`. This means that
parameters are provided on an individual basis for each element in the PyTree.

`SaveArgs` parameters include:

*   `aggregate`: if true, saves the given parameter to a unified msgpack
    checkpoint file. Must be false if the given array value is a sharded array.
*   `dtype`: if provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).

`RestoreArgs` parameters include:

*   `restore_type`: Specifies the type to restore the parameter as. This is
    necessary because many parameters can be restored as multiple different
    types. We have considered saving the type of a parameter as metadata
    alongside the value, so that the parameter can easily be restored as the
    type that it was originally saved as, but we currently feel that this may
    overly anchor a checkpoint to specific versions and specific type
    implementations.
*   `lazy`: if True, restores using [LazyValue](#LazyValue). The actual read
    operation will not be performed until `get` is called for the restored
    LazyValue
*   `dtype`: if provided, casts the parameter to the given dtype after
    restoring. Note that the parameter must be compatible with the given type
    (e.g. jnp.bfloat16 is not compatible with np.ndarray).

`RestoreArgs` is overridden by `ArrayRestoreArgs`, which should be used when
restoring a parameter as a sharded array. This class includes additional
parameters:

*   `mesh`: the device mesh that the array should be restored with. Cannot be
    None.
*   `mesh_axes`: the mesh_axes that the array should be restored with. Cannot be
    None.
*   `global_shapes`: the global shape that the array should be restored into. If
    not provided, the shape will be restored as written. Padding or truncation
    may occur if the provided shape does not match the saved shape.

`PyTreeCheckpointHandler` will create an individual directory for each nested
key. The exact naming of per-parameter directories can be customized by
overriding `_get_param_infos`. These parameters are saved using
[Tensorstore](https://google.github.io/tensorstore/). There is an additional
`checkpoint` file that will be created using the
[msgpack](https://msgpack.org/index.html) file format. This stores the PyTree
structure, as well as any parameters for which `aggregate` was True at save
time. Individual directories will not be created for these parameters.

For `restore`, `item` is an optional argument because the PyTree structure can
be recovered from the saved checkpoint if `item` is a dictionary. However, if
`item` is an object other than `dict`, `item` should be provided in order to
restore the object structure.

### JsonCheckpointHandler

[`JsonCheckpointHandler`](https://github.com/google/orbax/tree/main/orbax/checkpoint/json_checkpoint_handler.py)
is provided as a way to checkpoint nested dictionaries that can be serialized in
JSON format. This can be useful as a way to store checkpoint metadata. For
example, `CheckpointManager` uses this class to store the metrics used to
evaluate relative checkpoint quality.

Note that presently, this class does not implement async APIs.

## TypeHandler and AggregateHandler

These classes represent an internal implementation detail of
[`PyTreeCheckpointHandler`](#pytreecheckpointhandler) that allows for greater
customizability for PyTree leaf types and storage media. In the vast majority of
cases, users can safely ignore `TypeHandler` and `AggregateHandler`. However, if
a user has a custom object type or specific storage requirements, they may wish
to customize these classes.

### TypeHandler

[`TypeHandler`](https://github.com/google/orbax/tree/main/orbax/checkpoint/type_handlers.py)
and [`AggregateHandler`](#aggregatehandler) are classes designed specifically
for use with [`PyTreeCheckpointHandler`](#pytreecheckpointhandler) that grant
users greater flexibility when dealing with PyTrees with custom leaf types or
custom logic for saving these leaves.

For example, standard `TypeHandler` implementations and the types that they
handle include:

*   `ArrayHandler`: `jax.Array`
*   `NumpyHandler`: `np.ndarray`
*   `ScalarHandler`: `int`, `float`

These default implementations all use Tensorstore to serialize and deserialize
data.

For most users, `TypeHandler` is an internal implementation detail that need not
concern them. It will only become relevant if a user has a custom leaf type in a
PyTree that the wish to save and restore.

`TypeHandler` provides the following APIs:

*   `param_infos`: Constructs internal information (such as tensorstore spec)
    needed to save the value.
*   `serialize`: Writes the value to a storage location.
*   `deserialize`: Reads the value from a storage location.

### AggregateHandler

[`AggregateHandler`](https://github.com/google/orbax/tree/main/orbax/checkpoint/aggregate_handlers.py)
provides a means for saving multiple parameters in a PyTree into a single file,
potentially allowing for greater storage space savings. Like
[`TypeHandler`](#typehandler), this class is also designed for use with
`PyTreeCheckpointHandler`.

While `TypeHandler` is designed for use with a single value of a specific type,
`AggregateHandler` must, by definition, work with entire PyTrees. Leaves that
were already serialized individually (to Tensorstore, perhaps) using a
`TypeHandler` will be replaced with space-saving dummy values.

The default implementation of `AggregateHandler` is `MsgpackHandler`, which
serializes a PyTree into the msgpack format.

## Utilities

### LazyValue

[`LazyValue`](https://github.com/google/orbax/tree/main/orbax/checkpoint/lazy_array.py)
provides a mechanism for delayed loading of values from a checkpoint. If a
parameter is restored as a `LazyValue` (in `PyTreeCheckpointHandler`, setting
`RestoreArgs.lazy = True`), the restored object will not yet have done the work
of actually loading the parameter from Tensorstore.

The actual loading will only be done when `.get()` is called on the `LazyValue`.

Of course, for parameters aggregated into a single file containing many
parameters, the loading does happen eagerly, regardless of whether `LazyValue`
is used. However, parameters saved in this way should typically be small.

### Transformations

The
[`transform_utils`](https://github.com/google/orbax/tree/main/orbax/checkpoint/transform_utils.py)
library provides functions to allow structural PyTree transformations, which can
facilitate migrations between different checkpoint versions.

The API consists of a `Transform` class and an `apply_transformations` function.

#### Transform

`Transform` consists of the following elements:

*   `original_key`: Denotes the original name of the key. Represented as a
    string with '/' denoting successive levels of nesting. If the key
    corresponding to this Transform is a regex, backreferences (such as \1) will
    be replaced with the appropriate matched group in the regex. Note: not
    needed if multi_value_fn is provided.
*   `use_fallback`: if True, takes the value from the fallback tree. If
    `default_to_original=True` in `apply_transformations`, the fallback tree is
    `new_tree`. If `default_to_original=False` in `apply_transformations`, the
    fallback tree is `original_tree`.
*   `value_fn`: A function accepting a single value and returning a single
    value. The value provided as an argument is the value of the transformation
    key in the original PyTree.
*   `multi_value_fn`: A function accepting a PyTree and returning any value. The
    PyTree argument will be the original PyTree, and the function should return
    the value of the key in the new PyTree.

This can be represented with some simple examples:

`{'a': Transform(original_key='b')}`

This denotes that the original key was named 'b', but we are changing it to 'a'.

`{'a': Transform(value_fn=lambda kv: kv['b'] * 2)}`

This signifies that the new key 'a' is the old key 'b' multiplied by two.

`{r'(.*)a(.*)': Transform(original_key=r'\1b\2'}`

This denotes that keys containing 'b' should be substituted to 'a'. This may
apply to multiple different keys at different levels of nesting. The '/'
character denotes a successive level of nesting.

#### Using Transformations

The `apply_transformations` function accepts an original PyTree, a PyTree of
`Transform` objects and a "new" Pytree. The function will return a PyTree
matching `new_tree`.

For example:

```py
original_tree = {
  'a': 1,
  'b': {
    'c': 5,
    'd': [0, 1, 2, 3]
  },
  'f': 2,
  'b1': {
    'c': 2,
  },
  'b2': {
    'c': 3,
  },
}
transformations = {
  'a1': Transform(original_key='a'),  # rename
  # another way of doing above
  'a1': Transform(multi_value_fn=lambda kv: kv['a']),
  'b': {
    # doubled original
    'c': Transform(multi_value_fn=lambda kv: kv['b']['c'] * 2)
    # drop b/d
  },
  # Copy original into multiple new keys
  'c1': Transform(original_key='b/c'),
  'c2': Transform(original_key='b/c'),
  # one to many mapping
  'x': Transform(multi_value_fn=lambda kv: kv['b']['d'][0]),
  'y': Transform(multi_value_fn=lambda kv: kv['b']['d'][1:]),
  # many to one mapping
  'z': Transform(multi_value_fn=lambda kv: kv['a'] * 2 + sum(kv['b']['d'])),
  r'x(\d.*)': Transform(original_key=r'b\1')
}

# defines the structure of the result
new_tree = {
  'a1': ...,
  'a1': ...,
  'b': {
    'c': ...,
  },
  'c1': ...,
  'c2': ...,
  'x': ...,
  'y': ...,
  'z': ...,
  # 'f' defined in original_tree and new_tree, but not in transforms. Value
  # carried over from original_tree.
  'f': ...,
  # This value matters since it is not present in original_tree or
  # transformations, so the value here will simply be preserved in the result.
  'g': 5,
  # These are just 'b1', 'b2', but renamed to 'x1', 'x2', with all values
  # copied over.
  'x1': {
    'c': 2,
  }
  'x2': {
    'c': 3,
  }
}

transformed_tree = apply_transformations(original_tree, transforms, new_tree)
```

Note that there is an additional option for `apply_transformations`, which is
`default_to_original` (True by default). This means that the values keys
unspecified in `transformations` but present in *both* trees will be taken from
the *original* tree. If False, such values will be taken from the *new* tree.

Remember that if a key is present in the new tree, but not in the old, the value
will simply be taken from the new tree. If a key is present in the original tree
but not in the new, it will be dropped in the result.

#### Examples

Let's consider a real-world example. In this scenario, we have a saved
checkpoint with parameters `Dense_0`, `Dense_1`. We want to restore this
checkpoint, with modifications, into a model for training with layers `Dense_0`,
`Dense_1`, `Dense_2`, `Dense_3`.

In this example, we will map original layers 0 and 1 onto the new layers 1 and
2, respectively. We want the new layers 0 and 3 to be initialized randomly, or
with some new values.

The new model may be initialized as a Flax
[TrainState](https://flax.readthedocs.io/en/latest/flax.training.html#train-state),
for example.

```py
params = model.init(
    jax.random.PRNGKey(0), jnp.ones([1, model.input_size]))
new_state = TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer)
# Restore original state.
original_state = manager.restore(step)
```

```py
 transformations = {
      # NewModel layer 0 is a newly inserted layer, thus use_fallback=True.
      r'(.*)Dense_0(.*)': Transform(use_fallback=True),
      # OriginalModel layer 0 maps to NewModel layer 1
      r'(.*)Dense_1(.*)': Transform(original_key=r'\1Dense_0\2'),
      # OriginalModel layer 1 maps to NewModel layer 2
      r'(.*)Dense_2(.*)': Transform(original_key=r'\1Dense_1\2')
  }  # Note: NewModel layer 3 is newly added.
  restored_state = apply_transformations(original_state, transformations, new_state)
```

Let's unpack what's happening with these transformations.

For layer 0, we want to instruct the function to ignore what's in
`original_state`, and to instead use the value from `new_state`. For this, we
set `use_fallback=True`.

For `Dense_1` and `Dense_2`, we simple provide a regex mapping the original name
of the key (`Dense_0` and `Dense_1`, respectively) to their new values using the
`original_key` field. Note that we can use a regex to match any key containing
the desired pattern, since a PyTree checkpoint will typically represent a single
layer with multiple different arrays, each containing the pattern.

Finally, we can simply omit `Dense_3` from `transformations`, as the `Dense_3`
was provided as a key in `new_state` and the function will simply take the value
from `new_state` and put it in the result.

## Miscellaneous

### Distributed Initialization

Using Orbax requires initializing the JAX distributed system. In a single-host
environment, this can be done easily using the following:

```py
import jax
import portpicker

port = portpicker.pick_unused_port()
jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)
```

This is often appropriate for colabs or other simple setups.

In a multi-host environment, a coordinator must be started by calling the
following on every process. More details are provided in the JAX
[documentation](https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster).

```py
import jax

# IP address of primary host, unused port.
jax.distributed.initialize(
  coordinator_address="192.168.0.1:1234",
  num_processes=2,
  process_id=0
)
```

