General Utilities
======================================

.. currentmodule:: orbax.checkpoint.utils

.. automodule:: orbax.checkpoint.utils

Async wrappers
------------------------
.. autofunction:: async_makedirs
.. autofunction:: async_write_bytes
.. autofunction:: async_exists

Aggregate file
------------------------
.. autofunction:: leaf_is_placeholder
.. autofunction:: leaf_placeholder
.. autofunction:: name_from_leaf_placeholder
.. autofunction:: is_supported_empty_aggregation_type
.. autofunction:: is_supported_aggregation_type

Directories
------------------------
.. autofunction:: cleanup_tmp_directories
.. autofunction:: get_tmp_directory
.. autofunction:: create_tmp_directory
.. autofunction:: get_save_directory
.. autofunction:: is_gcs_path

Atomicity
------------------------
.. autofunction:: is_tmp_checkpoint
.. autofunction:: is_checkpoint_finalized

Checkpoint steps
------------------------
.. autofunction:: step_from_checkpoint_name
.. autofunction:: checkpoint_steps_paths
.. autofunction:: checkpoint_steps
.. autofunction:: any_checkpoint_step
.. autofunction:: tmp_checkpoints
.. autofunction:: lockdir
.. autofunction:: is_locked
.. autofunction:: are_locked

Sharding
------------------------
.. autofunction:: fully_replicated_host_local_array_to_global_array

Misc.
------------------------
.. autofunction:: is_scalar
.. autofunction:: record_saved_duration