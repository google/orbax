General Utilities
======================================

.. currentmodule:: orbax.checkpoint.utils

.. automodule:: orbax.checkpoint.utils

Aggregate file
------------------------
.. autofunction:: leaf_is_placeholder
.. autofunction:: leaf_placeholder
.. autofunction:: name_from_leaf_placeholder

Directories
------------------------
.. autofunction:: cleanup_tmp_directories
.. autofunction:: get_tmp_directory
.. autofunction:: create_tmp_directory
.. autofunction:: get_save_directory
.. autofunction:: is_gcs_path

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