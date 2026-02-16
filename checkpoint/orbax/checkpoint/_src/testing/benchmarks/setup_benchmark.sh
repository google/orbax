#!/bin/bash

# Define the flag check
MOUNT_GCS=false
for arg in "$@"; do
  if [ "$arg" == "--mount" ] || [ "$arg" == "-m" ]; then
    MOUNT_GCS=true
  fi
done

# 1. Unmount and clean up (Always runs to ensure a clean state)
echo "Cleaning up /gcs..."
sudo fusermount -u /gcs 2>/dev/null || echo "/gcs not mounted"
sudo rm -rf /gcs

# 2. Drop Caches
echo "Dropping system caches..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
free -h

# 3. Conditional GCS Mount
if [ "$MOUNT_GCS" = true ]; then
    echo "Recreating /gcs mount point..."
    sudo mkdir -p /gcs
    sudo chown $USER:$USER /gcs
    sudo chmod 755 /gcs

    echo "Mounting GCS bucket..."
    gcsfuse --implicit-dirs /gcs
    ls -la /gcs/safetensor-kimi-central/
else
    echo "Skipping GCS mount step (use --mount to enable)."
fi

echo "Resetting Orbax repository..."
cd ~
rm -rf orbax
git clone https://github.com/google/orbax.git
cd orbax
git fetch origin pull/2935/head:pr-2935
git checkout pr-2935

echo "Setup complete."