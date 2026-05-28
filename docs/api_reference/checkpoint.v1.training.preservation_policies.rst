``ocp.v1.training.preservation_policies`` module
============================================================================

.. currentmodule:: orbax.checkpoint.v1.training.preservation_policies

.. automodule:: orbax.checkpoint.experimental.v1.training.preservation_policies


PreservationPolicy
------------------------------------------------------------
.. autoclass:: PreservationPolicy
  :members: should_preserve
  :show-inheritance:

PreservationContext
----------------------------------------------------------
.. autoclass:: PreservationContext
  :members:

PreserveAll
----------------------------------------------------------
.. autoclass:: PreserveAll
  :members: should_preserve

LatestN
----------------------------------------------------------
.. autoclass:: LatestN
  :members: should_preserve

EveryNSeconds
----------------------------------------------------------
.. autoclass:: EveryNSeconds
  :members: should_preserve

EveryNSteps
----------------------------------------------------------
.. autoclass:: EveryNSteps
  :members: should_preserve

CustomSteps
----------------------------------------------------------
.. autoclass:: CustomSteps
  :members: should_preserve

BestN
----------------------------------------------------------
.. autoclass:: BestN
  :members: should_preserve

AnyPreservationPolicy
----------------------------------------------------------
.. autoclass:: AnyPreservationPolicy
  :members: should_preserve
