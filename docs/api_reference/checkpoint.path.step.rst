Step entities
======================================

.. currentmodule:: orbax.checkpoint.path.step

.. automodule:: orbax.checkpoint.path.step

Metadata
------------------------------
.. autoclass:: Metadata
  :members:

NameFormat
---------------------------------
.. autoclass:: NameFormat
  :members:

Factories for various NameFormats
---------------------------------
.. autofunction:: standard_name_format
.. autofunction:: composite_name_format

Helper functions
-----------------------
.. autofunction:: build_step_path
.. autofunction:: checkpoint_steps_paths
.. autofunction:: checkpoint_steps
.. autofunction:: any_checkpoint_step
.. autofunction:: latest_step_metadata
.. autofunction:: step_metadata_of_checkpoint_path

Helper functions (experimental)
-----------------------
.. autofunction:: find_step_path
.. autofunction:: is_gcs_path
.. autofunction:: get_save_directory
.. autofunction:: is_tmp_checkpoint
.. autofunction:: is_checkpoint_finalized
.. autofunction:: create_tmp_directory
.. autofunction:: tmp_checkpoints
.. autofunction:: cleanup_tmp_directories
.. autofunction:: get_tmp_directory
.. autofunction:: step_from_checkpoint_name

Helper functions to implement NameFormat classes
------------------------------------------------
.. autofunction:: build_step_metadatas
.. autofunction:: step_prefix_with_underscore