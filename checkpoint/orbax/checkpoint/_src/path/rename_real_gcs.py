print("--- Script starting now ---")

import types
from orbax.checkpoint import CheckpointManager
from etils import epath
import os
# Set logging levels to see detailed GCS API logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '5'

# --- Step 1: Verify the import ---
try:
    from deleter import StandardCheckpointDeleter
    # Extract the unbound method from the class
    _gcs_rename_step = StandardCheckpointDeleter._gcs_rename_step
    print("✅ SUCCESS: Imported '_gcs_rename_step' from deleter.py.")
except ImportError:
    print("❌ ERROR: Could not import from 'deleter.py'.")
    print("   Please ensure 'deleter.py' is in the same directory and contains the method.")
    exit() # Stop if the import fails

# --- Configuration ---
#BUCKET_NAME = "axlearn-nonhns-southamerica-west1"
#BASE_DIR = f"gs://{BUCKET_NAME}/nonhns-direct-500mb/checkpoints/step_00007840"

BUCKET_NAME = "axlearn-checkpoint-southamerica-west1"
BASE_DIR = f"gs://{BUCKET_NAME}/deepikarajani/deepikarajani-validate-3-tf-tpu/checkpoints/step_00000550"
#SOURCE_FOLDER_NAME = "source"
TRASH_SUBDIR = "_trash/"

def execute_rename():
    manager = CheckpointManager(directory=BASE_DIR)
    manager._todelete_full_path = TRASH_SUBDIR

    # --- Step 2: Verify the attachment ---
    print("\n--- Checking manager object ---")
    print(f"Before attach: Does manager have the method? {hasattr(manager, '_gcs_rename_step')}")

    # Dynamically attach your method from deleter.py to the instance.
    manager._gcs_rename_step = types.MethodType(_gcs_rename_step, manager)

    print(f"After attach:  Does manager have the method? {hasattr(manager, '_gcs_rename_step')}")
    print("-----------------------------\n")

    # --- Step 3: Call the method ---
    if not hasattr(manager, '_gcs_rename_step'):
        print("❌ CRITICAL ERROR: Failed to attach the method to the manager object. Cannot proceed.")
        return

    source_path = epath.Path(BASE_DIR) 
    print(f"Attempting to rename: {source_path}")
    print("...")

    try:
        manager._gcs_rename_step(step=550, delete_target=source_path)
        print("✅ Rename operation completed successfully.")
        print(f"Please check the '{TRASH_SUBDIR}' folder in your bucket to verify.")
    except Exception as e:
        print(f"❌ An error occurred during the call: {e}")
        print("Please ensure the source folder exists and you have correct permissions.")

if __name__ == "__main__":
    execute_rename()