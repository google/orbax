# Common V1 Errors & Troubleshooting

Quick remedies for common Orbax V1 checkpointing errors and distributed training
issues. For interactive inspection recipes, see the
[Debugging Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/debug_guide.html).

## Common Errors

### 1. `InvalidLayoutError`

```python
# ocp.errors.InvalidLayoutError: Could not recognize the checkpoint at ... as a valid Orbax checkpoint.
```

**Cause:** `path` points to a training root directory (`0/`, `100/`), an
incomplete `*.orbax-checkpoint-tmp*` directory, a checkpointable subdirectory
(`.../100/state`), or a non-Orbax format (e.g. Safetensors) without
`ctx.checkpoint_layout` set.

```python
# Load a specific step or use Checkpointer on a root directory:
state = ocp.load(root_dir / '100', abstract_state)

# Or configure layout for non-Orbax formats:
ctx = ocp.Context()
ctx.checkpoint_layout = ocp.options.CheckpointLayout.SAFETENSORS  # or AUTO
with ctx:
  state = ocp.load(safetensors_path, abstract_state)
```

See the
[Checkpoint Format Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/checkpoint_format.html).

### 2. Checkpointable Name Mismatch

```python
# ocp.errors.InvalidLayoutError: Checkpointable 'state' not found in checkpoint metadata.
```

**Cause:** `ocp.training.Checkpointer.load` defaults strictly to
`checkpointable_name='state'`, while `ocp.load` defaults to `'AUTO'` (which may
select an unintended PyTree when multiple exist).

```python
print(list(ocp.checkpointables_metadata(path).metadata.keys()))
state = ocp.load(path, abstract_state, checkpointable_name='model')
items = ocp.load_checkpointables(
    path, {'model': abstract_model, 'dataset': dataset_iter}
)
```

See the
[Checkpointables Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/checkpointables.html).

### 3. Sharding & Topology Mismatch

```python
# Topology mismatch when loading without an abstract target:
# ValueError: sharding passed to deserialization should be specified, concrete and an instance of jax.sharding.Sharding. Got None
# Missing `_sharding` file (e.g. enable_write_sharding_file=False) or `sharding=None` in target:
# ValueError: Sharding of jax.Array cannot be None. Provide mesh and mesh_axes OR sharding
```

**Cause:** Restoring a `jax.Array` requires a concrete `jax.sharding.Sharding`
valid for the currently attached devices; loading without one fails across
different topologies or missing `_sharding` files.

```python
# Pass an explicit target with NamedSharding on the current mesh:
target = jax.tree.map(
    lambda m: jax.ShapeDtypeStruct(m.shape, m.dtype, sharding=new_sharding)
    if hasattr(m, 'sharding_metadata') else m,
    ocp.metadata(path).metadata,
)
state = ocp.load(path, target)

# Or load a leaf into host RAM without accelerators:
ctx = ocp.Context()
ctx.pytree.loading.partial_load = True
with ctx:
  leaf = ocp.load(path, {'a': {'x': np.ndarray}})
```

See
[Loading with Resharding](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/maximizing_performance.html#loading-with-resharding).

### 4. Shape or PyTree Structure Mismatch

```python
# ValueError: ... shape mismatch / missing keys between target and checkpoint.
```

**Cause:** Array shapes or dictionary keys in `abstract_state` differ from the
saved checkpoint.

```python
ctx = ocp.Context()
ctx.array.loading.enable_padding_and_truncation = True  # Pad/truncate changed shapes.
ctx.pytree.loading.partial_load = True                  # Ignore omitted keys.
with ctx:
  state = ocp.load(path, target_abstract_state)
```

See
[Model Surgery (`ocp.PLACEHOLDER`)](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/model_surgery.html#placeholder).

### 5. Active Context Mutation & Thread Safety

```python
# RuntimeError: Cannot mutate options of an active context. Configure before entering the `with` block.
```

**Cause:** An `ocp.Context` is frozen while inside `with ctx:`, and Orbax APIs
must be invoked from the main thread.

```python
with parent_ctx:
  child_ctx = ocp.Context(parent_ctx)
  child_ctx.asynchronous.timeout_secs = 600
  with child_ctx:
    ocp.save(other_path, state)
```

See
[Configuring Specialized Features (`Context`)](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/context.html).

### 6. Training Step Errors

```python
# ocp.training.errors.StepAlreadyExistsError: Step 100 already exists.
# ocp.training.errors.StepNotFoundError: No checkpoint found ...
# ocp.errors.DeletionInProgressError: ...
```

**Cause:** `ocp.training.Checkpointer` expects strictly increasing steps;
resuming at `ckptr.latest.step` instead of `ckptr.latest.step + 1` raises
`StepAlreadyExistsError`, and older steps are skipped unless `force=True` /
`overwrite=True`.

```python
with ocp.training.Checkpointer(root_dir, cleanup_tmp_directories=True) as ckptr:
  if ckptr.latest is not None:
    state = ckptr.load(ckptr.latest, abstract_state=abstract_state)
    start_step = ckptr.latest.step + 1  # Resume at latest.step + 1, not latest.step.
  else:
    start_step = 0
  # Call ckptr.reload() if another process modified root_dir.
```

See the
[Training Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/training.html).

---

## Distributed Checkpointing FAQ

### Multi-Host Coordination

Call `jax.distributed.initialize(...)` on every host before Orbax, invoke all
`ocp.*` and `Checkpointer` calls in the same order on every active process
(never gate behind `jax.process_index() == 0`), and keep `ocp.Context` identical
across hosts. Use `ctx.multiprocessing` (`primary_host`, `active_processes`,
`barrier_sync_key_prefix`) for process subsets or concurrent checkpointers; see
[`Context`](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/context.html).

### Barrier Timeouts (`DEADLINE_EXCEEDED` / `TimeoutError`)

Barrier timeouts typically mean one host failed or diverged earlier (inspect
`[process=<id>]` logs across *all* hosts) or storage is in a different region.
Increase `ctx.asynchronous.timeout_secs` (default `1200`, must be a positive
`int`; `None` is not supported—it raises `TypeError` on `Checkpointer` and drops
`ocp.save*` to 300 s), and assign distinct
`ctx.multiprocessing.barrier_sync_key_prefix` values for concurrent
checkpointers.

### Host RAM or Device HBM OOM

Bound concurrent I/O and D2H buffers via `ctx.memory` (`read_concurrent_bytes`,
`write_concurrent_bytes`, `transfer_concurrent_bytes`, `regulator_options`), or
enable `ctx.array.loading.use_load_and_broadcast = True` for multi-replica
loads. See
[Maximizing Performance](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/maximizing_performance.html).

### Slow Resharding Across Topologies

Subdivide saved chunks at save time with
`ctx.array.saving.storage_options.chunk_byte_size` (typically ≥ 10 MiB) and
optional `shard_axes` so target shards align with saved chunks. See
[Loading with Resharding](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/maximizing_performance.html#loading-with-resharding).

### `ocdbt.process_<id>/` Subdirectories

The merged `manifest.ocdbt` references data files inside every
`ocdbt.process_<id>/` subdirectory, and V1 checkpoints require top-level
`orbax.checkpoint` and `_CHECKPOINT_METADATA`. Always copy or archive the entire
step directory intact; see the
[Checkpoint Format Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/checkpoint_format.html).

### Guaranteeing Async Saves Finish & Surfacing Errors

Retain the `AsyncResponse` returned by `ocp.save_async` or `ckptr.save_async`
and call `.result()` to reap background runner threads and surface exceptions;
always wrap `Checkpointer` in `with` (or call `ckptr.close()`).

```python
pending = None
with ocp.training.Checkpointer(root_dir) as ckptr:
  for step in range(start_step, num_steps):
    if pending is not None and not ckptr.is_saving_in_progress():
      pending.result()
      pending = None
    response = ckptr.save_async(step, state)
    if response is not None:
      if pending is not None:
        pending.result()
      pending = response
  if pending is not None:
    pending.result()
```

See the
[Asynchronous Checkpointing Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/async_checkpointing.html).

### Preemption Recovery & Storage Formats

Use `ocp.training.save_decision_policies` (`ContinuousCheckpointingPolicy` or
`PreemptionCheckpointingPolicy`) with `cleanup_tmp_directories=True` on
`Checkpointer`, and keep the default storage format (`use_ocdbt=True`,
`use_zarr3=True`, `use_compression=True`) in production. See the
[Training Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/training.html)
and
[Maximizing Performance](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/maximizing_performance.html).

### Filing a Bug Report

File issues on the [Orbax issue tracker](https://github.com/google/orbax/issues)
with package versions (`orbax-checkpoint`, `jax`, `jaxlib`, `tensorstore`),
device topology/host count, storage backend and non-default `ocp.Context`
options, `[process=<id>]` logs from the *first* failing host, and a minimal
reproducer.
