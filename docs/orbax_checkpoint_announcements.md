# Announcements

## 2024-02-06
Custom logging configuration was broken from `0.5.0` to `0.5.2`. It is fixed in
`0.5.3`. Please see https://github.com/google/orbax/issues/699 for details.

## 2024-02-01
`SaveArgs.aggregate` is deprecated. Please use
 [custom TypeHandler](https://orbax.readthedocs.io/en/latest/custom_handlers.html#typehandler)
  or contact Orbax team to learn more. Please migrate before **May 1st, 2024**.

## 2024-01-18
`CheckpointManager.save(...)` is now async by default. Make sure you call
 `wait_until_finished` if depending on a previous save being completed.
Otherwise, the behavior can be disabled via the
`CheckpointManagerOptions.enable_async_checkpointing` option.


## 2024-01-12
The `CheckpointManager` API is changing. Please see the
[migration instructions](https://orbax.readthedocs.io/en/latest/orbax_checkpoint_101.html)
and complete your migration by **May 1st, 2024**.
