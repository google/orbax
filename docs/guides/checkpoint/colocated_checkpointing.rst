
Colocated Python Checkpointing
================================

.. warning::
    This is an experimental feature and requires a Pathways environment to run.

Design
------

The ColocatedPythonArrayHandler is an experimental feature that provides a more performant way to save and restore checkpoints in a multi-host, multi-device setting. It leverages Pathways-specific infrastructure to ensure that the checkpointing operations happen on the same devices as the data, avoiding unnecessary data transfer.

How to use the colocated_python annotation
------------------------------------------

To use the ColocatedPythonArrayHandler, you need to annotate your JAX arrays with ``ocp.args.StandardSave(..., colocated=True)``. This tells Orbax to use the ColocatedPythonArrayHandler for that specific array.

.. code-block:: python

    import orbax.checkpoint as ocp
    import jax.numpy as jnp

    # ...

    checkpointer = ocp.PyTreeCheckpointer()
    data = {'my_array': jnp.ones((10,))}
    save_args = ocp.args.StandardSave(colocated=True)
    checkpointer.save(path, args=ocp.args.PyTreeSave(tree=data, save_args=save_args))

Example usage in memory
-----------------------

Here's a complete in-memory example of how to use the ColocatedPythonArrayHandler:

.. code-block:: python

    import orbax.checkpoint as ocp
    from orbax.checkpoint.experimental import ColocatedPythonArrayHandler
    import jax
    import jax.numpy as jnp
    import numpy as np
    from etils import epath

    # Register the handler
    ocp.register_type_handler(jax.Array, ColocatedPythonArrayHandler(), override=True)

    class Example:
        def __init__(self):
            self.checkpointer = ocp.PyTreeCheckpointer()
            self.path = epath.Path('/tmp/orbax-checkpoint')

        def save(self, data):
            save_args = ocp.args.StandardSave(colocated=True)
            self.checkpointer.save(self.path, args=ocp.args.PyTreeSave(tree=data, save_args=save_args))

        def restore(self):
            return self.checkpointer.restore(self.path)

    # Create some data
    data = {'x': jnp.ones((10,)), 'y': np.arange(5)}

    # Save the data
    example = Example()
    example.save(data)

    # Restore the data
    restored_data = example.restore()

    # Verify the data
    np.testing.assert_array_equal(restored_data['x'], data['x'])
    np.testing.assert_array_equal(restored_data['y'], data['y'])

    print("Checkpoint saved and restored successfully!")
