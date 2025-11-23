import jax
import jax.numpy as jnp
import pytest

from orbax.checkpoint.metadata.tree import (
    KeyMetadataEntry,
    NestedKeyMetadataEntry
)
from orbax.checkpoint.metadata import tree_utils


class TestExportTypeConversions:
    """
    Integration test using real JAX PyTrees.
    Ensures that numeric array-axis keys serialize and deserialize correctly.
    """

    def test_export_type_conversions(self):
        """
        Create a JAX PyTree, extract key metadata, serialize and deserialize,
        and ensure numeric string keys are converted back into integers.
        """
        # Create a sample JAX structure
        pytree = {
            "layer": [
                jnp.array([1, 2, 3]),   # index 0 → numeric key
                jnp.array([4, 5, 6])    # index 1 → numeric key
            ]
        }

        # Build the keypath from the pytree
        keypaths = list(tree_utils.flatten_with_path(pytree))
        keypath = keypaths[0][0]  # take first path e.g. ("layer", 0)

        # Build metadata entry for that keypath
        metadata_entry = KeyMetadataEntry.build(keypath)

        # Serialize to JSON
        json_data = metadata_entry.to_json()

        # Deserialize back
        restored = KeyMetadataEntry.from_json(json_data)

        # Extract nested entry for the second level (numeric key index)
        numeric_entry: NestedKeyMetadataEntry = restored.nested_key_metadata_entries[1]

        # Assertions validating type conversion behavior
        assert isinstance(numeric_entry.nested_key_name, int)
        assert numeric_entry.nested_key_name == 0