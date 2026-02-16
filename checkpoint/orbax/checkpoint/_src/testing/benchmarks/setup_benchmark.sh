#!/bin/bash

# 1. Unmount and clean up (Always runs to ensure a clean state)
echo "Cleaning up /gcs..."
sudo fusermount -u /gcs 2>/dev/null || echo "/gcs not mounted"
sudo rm -rf /gcs

# 2. Drop Caches
echo "Dropping system caches..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# 3. GCS Mount
echo "Installing gcsfuse dependencies..."
sudo apt-get update -qq > /dev/null 2>&1
sudo apt-get install -qq -y curl gnupg lsb-release > /dev/null 2>&1

echo "Installing gcsfuse..."
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - > /dev/null 2>&1
sudo apt-get update -qq > /dev/null 2>&1
sudo apt-get install -qq -y gcsfuse > /dev/null 2>&1

echo "Recreating /gcs mount point..."
sudo mkdir -p /gcs
sudo chown $USER:$USER /gcs
sudo chmod 755 /gcs

echo "Mounting GCS bucket..."
gcsfuse --implicit-dirs /gcs
ls -la /gcs/safetensor-kimi-central/

echo "Installing git..."
sudo apt-get update -qq > /dev/null 2>&1
sudo apt-get install -qq -y git > /dev/null 2>&1
echo "Git installed."

echo "Resetting Orbax repository..."
cd ~
rm -rf orbax
git clone https://github.com/google/orbax.git > /dev/null 2>&1
cd orbax > /dev/null 2>&1
git fetch origin pull/2935/head:pr-2935 > /dev/null 2>&1
git checkout pr-2935 > /dev/null 2>&1
free -h

echo "Setup complete. we are at $(pwd)"