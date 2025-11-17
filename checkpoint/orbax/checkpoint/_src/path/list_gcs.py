print("--- Script starting now ---")

import types
from orbax.checkpoint import CheckpointManager, path as ocp_path
from etils import epath
import os

# --------------------------------------------------------------------------
# ACTION REQUIRED:
# 1. Save your custom _glob_step_paths function in a file (e.g., `my_functions.py`).
# 2. Update the import statement below to match your filename.
# --------------------------------------------------------------------------
try:
    from step import _glob_step_paths
    print("✅ SUCCESS: Imported custom '_glob_step_paths' function.")
except ImportError:
    print("❌ ERROR: Could not import from your file.")
    print("   Please ensure the file with your function is in the same directory.")
    exit()

# --- Configuration ---
BUCKET_NAME = "axlearn-nonhns-southamerica-west1"
# The base path should be the directory CONTAINING the steps.
BASE_DIR = f"gs://{BUCKET_NAME}/nonhns-direct-500mb/checkpoints/"

# This helper function is likely used inside your _glob_step_paths implementation.
step_prefix_with_underscore = ocp_path.step_prefix_with_underscore

def test_glob_function():
    """
    Attaches and calls the _glob_step_paths method, then prints the results.
    """
    manager = CheckpointManager(directory=BASE_DIR)
    
    # The glob pattern relies on `self.step_prefix`. We must set it.
    manager.step_prefix = 'step'

    # --- Attach the custom method ---
    # This is the crucial step we need to add back in.
    print("\n--- Attaching custom method to manager instance ---")
    manager._glob_step_paths = types.MethodType(_glob_step_paths, manager)
    print(f"Method attached: {hasattr(manager, '_glob_step_paths')}")

    print("\n--- Calling _glob_step_paths ---")
    path_to_scan = epath.Path(BASE_DIR)
    print(f"Attempting to find step paths in: {path_to_scan}")
    print("...")

    try:
        # Now this call will work because the method exists on the instance.
        found_paths = manager._glob_step_paths(base_path=path_to_scan)

        # Process and display the results.
        print(f"\n✅ SUCCESS: _glob_step_paths completed.")
        if found_paths:
            print(f"Found {len(found_paths)} step paths:")
            for path in found_paths:
                print(f"  - {path}")
        else:
            print("Found 0 step paths. Ensure the directory is not empty and the prefix is correct.")

    except Exception as e:
        print(f"❌ An error occurred during the call: {e}")

if __name__ == "__main__":
    test_glob_function()