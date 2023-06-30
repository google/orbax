General Utilities
======================================

.. currentmodule:: orbax.checkpoint.utils

.. automodule:: orbax.checkpoint.utils

Device sync
------------------------
.. autofunction:: should_skip_device_sync
.. autofunction:: sync_global_devices
.. autofunction:: broadcast_one_to_all

Async wrappers
------------------------
.. autofunction:: async_makedirs
.. autofunction:: async_write_bytes
.. autofunction:: async_exists

Tree utils
------------------------
.. autofunction:: is_empty_or_leaf
.. autofunction:: get_key_name
.. autofunction:: to_flat_dict
.. autofunction:: serialize_tree
.. autofunction:: deserialize_tree
.. autofunction:: from_flat_dict
.. autofunction:: pytree_structure

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
.. autofunction:: ensure_atomic_save
.. autofunction:: on_commit_callback
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