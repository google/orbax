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

Convenience functions
-----------------------
.. autofunction:: build_step_path

Helper functions to implement NameFormat classes
------------------------------------------------
.. autofunction:: build_step_metadatas
.. autofunction:: step_prefix_with_underscore