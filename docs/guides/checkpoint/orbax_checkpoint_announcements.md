# Announcements

## 2024-04-02
The `checkpoint_manager_context(...)` function is deprecated. To ensure proper 
resource handling, please update your code to use either 
`with CheckpointManager(...) as manager:` or explicitly call `manager.close()` 
before your program exits. This will prevent incomplete background operations 
such as deleting old checkpoints.

## 2024-02-01
`SaveArgs.aggregate` is deprecated. Please use
 [custom TypeHandler](https://orbax.readthedocs.io/en/latest/custom_handlers.html#typehandler)
  or contact Orbax team to learn more. Please migrate before **August 1st, 2024**.

## 2024-01-18
`CheckpointManager.save(...)` is now async by default. Make sure you call
 `wait_until_finished` if depending on a previous save being completed.
Otherwise, the behavior can be disabled via the
`CheckpointManagerOptions.enable_async_checkpointing` option.


## 2024-01-12
The `CheckpointManager` API is changing. Please see the
[migration instructions](https://orbax.readthedocs.io/en/latest/orbax_checkpoint_101.html)
and complete your migration by **August 1st, 2024**.
