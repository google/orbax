#!/bin/bash
# setup_tpu.sh
# Usage: setup_tpu.sh [REPO_URL] [BRANCH_OR_PR]
# Example: setup_tpu.sh https://github.com/google/orbax.git main
# Example: setup_tpu.sh https://github.com/google/orbax.git refs/pull/123/head

set -e

# Default values
REPO_URL="https://github.com/google/orbax.git"
BRANCH="main"
JAX_VERSION="newest"
PR_NUMBER=""
RAMFS_DIR=""

# Parse flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --repo-url) REPO_URL="$2"; shift ;;
        --branch) BRANCH="$2"; shift ;;
        --pr) PR_NUMBER="$2"; shift ;;
        --jax-version) JAX_VERSION="$2"; shift ;;
        --ramfs-dir) RAMFS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

WORK_DIR="/app/orbax_repo"

echo "=== Starting TPU VM Setup ==="
# Info
cat /etc/os-release


# Wait for apt lock to be released (unattended-upgrades often runs on boot)
# This prevents "E: Could not get lock /var/lib/dpkg/lock-frontend" errors.
echo ">>> Waiting for apt lock..."
while sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1 || \
      sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 || \
      sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "Waiting for apt lock..."
    sleep 2
done
echo ">>> Apt lock acquired."

echo "Repo: $REPO_URL"
echo "Branch/Ref: $BRANCH"

# Retry loop for apt-get to be robust against transient locks
max_retries=5
for i in $(seq 1 $max_retries); do
    echo ">>> Attempting setup (Attempt $i/$max_retries)..."
    if sudo apt-get update && sudo apt-get install -y python3-pip git dnsutils; then
        echo ">>> System dependencies installed."
        break
    else
        echo ">>> Apt failed. Retrying in 5s..."
        sleep 5
        if [ $i -eq $max_retries ]; then
            echo "Error: Apt setup failed after $max_retries attempts."
            exit 1
        fi
    fi
done

# 2. Python Setup
echo ">>> Using System Python 3..."
python3 --version

# Ensure pip is up to date
echo ">>> Upgrading pip..."
pip install --upgrade pip

# 3. Clone/Fetch Orbax
echo ">>> Setting up Orbax workspace at $WORK_DIR..."

if [ -d "$WORK_DIR" ]; then
    echo "Directory exists, clearing..."
    TRASH_DIR="${WORK_DIR}_trash_$(date +%s)"
    sudo mv "$WORK_DIR" "$TRASH_DIR"
    # Try to remove trash in background to not block setup
    (sudo rm -rf "$TRASH_DIR" || true) &
fi

sudo mkdir -p "$WORK_DIR"
sudo chown -R "$USER:$USER" "$WORK_DIR"

cd "$WORK_DIR"
git init
git remote add origin "$REPO_URL"

if [ -n "$PR_NUMBER" ]; then
    echo "Fetching PR #${PR_NUMBER} (Shallow)..."
    git fetch --depth 1 origin pull/$PR_NUMBER/head:pr_branch
    git checkout pr_branch
else
    echo "Fetching branch: ${BRANCH} (Shallow)..."
    git fetch --depth 1 origin "$BRANCH"
    git checkout FETCH_HEAD
fi

# 4. Install Python Dependencies
echo ">>> Installing Python dependencies..."

# Install JAX (Flexible Versions matching Dockerfile)
echo ">>> Installing JAX (Version: $JAX_VERSION)..."

if [ "$JAX_VERSION" = "newest" ]; then
    # TPU default
    pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
elif [ "$JAX_VERSION" = "nightly" ]; then
    # TPU nightly
    pip install -U --pre "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
        --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/
else
    # Specific version
    pip install "jax[tpu]==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

# Install Orbax from source
echo "Installing Orbax from source..."
# Navigate to checkpoint subdirectory if it exists (Orbax repo structure)
if [ -d "checkpoint" ]; then
    cd checkpoint
fi
# Fix for some build isolation issues with old pip/setuptools
pip install .

# Install Benchmark dependencies (from Dockerfile analysis)
echo "Installing benchmark dependencies..."
python3 -m pip install gcsfs portpicker clu tensorflow google-cloud-logging

if [ -n "$RAMFS_DIR" ]; then
    echo ">>> Setting up high-performance storage..."
    # 1. Define the mount point
    sudo mkdir -p "${RAMFS_DIR}"

    # 2. Mount tmpfs
    if ! grep -qs " ${RAMFS_DIR} " /proc/mounts; then
        echo ">>> Mounting tmpfs for high-performance storage..."
        # Options:
        #   -t tmpfs: Specifies the filesystem type as tmpfs.
        #   -o size=32g: Sets a maximum size for the tmpfs. Adjust "32g" based on your TPU VM's RAM.
        #                 It's crucial to leave enough RAM for the OS and your applications.
        #   -o rw: Read-write permissions.
        #   -o huge=always: (Optional) Encourage huge pages for potentially better performance with large files.
        #   -o mode=1777: Standard permissions for shared temporary directories.
        #   -o relatime: Update access times relative to modification time.
        sudo mount -t tmpfs -o size=32g,rw,huge=always,mode=1777,relatime tmpfs "${RAMFS_DIR}"
        echo ">>> tmpfs mounted at ${RAMFS_DIR} with size limit of 32GB."
    else
        echo ">>> ${RAMFS_DIR} is already a mountpoint, skipping mount."
    fi

    echo ">>> Deleting existing directories under ${RAMFS_DIR}..."
    sudo rm -rf "${RAMFS_DIR}/*"

    # 3. Set ownership to the current user so your scripts can write to it
    sudo chown "${USER}:${USER}" "${RAMFS_DIR}"

    df -h "${RAMFS_DIR}"
fi

echo "=== Setup Complete ==="
echo "Orbax installed in: $WORK_DIR"
