#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# 1. Default Configuration (Edit these if needed)
# ==========================================
CLUSTER="orbax-safetensor-cpu-cluster"
PROJECT="orbax-checkpoint"
ZONE="us-central1-a"
INPUT_DIR="gs://safetensor-kimi-central/test-model-kimi"
OUTPUT_DIR="gs://safetensor-kimi-central/tmp/orbax-benchmark"
# Change gcr.io to us-central1-docker.pkg.dev if you are using Artifact Registry
IMAGE_REGISTRY="gcr.io/${PROJECT}/orbax-safetensor-cpu" 

# Initialize empty variables
PR_NUMBER=""
MACHINE_NAME="n2-standard-32"
MACHINE_COUNT="8"
DELETE_CLUSTER=0

# ==========================================
# 2. Parse Command Line Arguments
# ==========================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pr) PR_NUMBER="$2"; shift ;;
        --machine-name) MACHINE_NAME="$2"; shift ;;
        --machine-count) MACHINE_COUNT="$2"; shift ;;
        --cluster) CLUSTER="$2"; shift ;;
        --project) PROJECT="$2"; shift ;;
        --zone) ZONE="$2"; shift ;;
        --input-dir) INPUT_DIR="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --delete-cluster) DELETE_CLUSTER=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$DELETE_CLUSTER" -eq 1 ]; then
    echo "🔥 Installing xpk..."
    python3 -m venv /tmp/xpk-venv
    source /tmp/xpk-venv/bin/activate
    pip install -qq xpk

    echo "🔥 Deleting xpk cluster..."
    xpk cluster delete \
      --cluster="${CLUSTER}" \
      --project="${PROJECT}" \
      --zone="${ZONE}" \
      --force
    echo "✅ Cluster deleted."
    exit 0
fi

# Ensure the PR number was provided
if [ -z "$PR_NUMBER" ]; then
    echo "❌ Error: --pr is required (e.g., --pr 2347)"
    echo "Usage: ./launch_cluster.sh --pr 2347 --machine-name n2-standard-32 --machine-count 8"
    echo "To delete cluster: ./launch_cluster.sh --delete-cluster"
    exit 1
fi

# ==========================================
# 3. Setup Variables
# ==========================================
DEVICE_TYPE="${MACHINE_NAME}-${MACHINE_COUNT}"
IMAGE_TAG="${IMAGE_REGISTRY}:pr-${PR_NUMBER}"
# Add a timestamp so you can run the same PR multiple times without a name collision
WORKLOAD_NAME="orbax-pr-${PR_NUMBER}-$(date +%s)" 

echo "========================================"
echo "🚀 Starting Orbax Workload Pipeline"
echo " PR Number    : $PR_NUMBER"
echo " Device Type  : $DEVICE_TYPE"
echo " Image Tag    : $IMAGE_TAG"
echo " Workload Name: $WORKLOAD_NAME"
echo "========================================"

# ==========================================
# 4. Generate the Dockerfile dynamically
# ==========================================
echo "📝 Creating Dockerfile..."
cat << 'EOF' > Dockerfile
FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV JAX_PLATFORMS=cpu

RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends git curl gnupg lsb-release ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends gcsfuse && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -qq absl-py etils huggingface_hub "jax[cpu]" numpy tensorflow-cpu google-cloud-storage flax kubernetes
WORKDIR /workspace
RUN git clone https://github.com/google/orbax.git

WORKDIR /workspace/orbax
# The ARG command lets us inject the PR number from the bash script!
ARG PR_NUM
RUN git fetch origin pull/${PR_NUM}/head:pr-${PR_NUM} && \
    git checkout pr-${PR_NUM} && \
    pip install -qq -e ./checkpoint
EOF

# ==========================================
# 5. Build and Push Docker Image
# ==========================================
echo "🐳 Building Docker image..."
# Pass the PR_NUMBER into the Dockerfile using --build-arg
docker build --no-cache --build-arg PR_NUM="${PR_NUMBER}" -t "${IMAGE_TAG}" .

echo "☁️ Pushing Docker image to Google Cloud..."
docker push "${IMAGE_TAG}"

# ==========================================
# 6. Launch xpk workload
# ==========================================
echo "🔥 Installing xpk and gke-gcloud-auth-plugin..."
python3 -m venv /tmp/xpk-venv
source /tmp/xpk-venv/bin/activate
pip install -qq xpk

# Ensure gke-gcloud-auth-plugin is installed for kubectl authentication
sudo apt-get update -qq
sudo apt-get install -qq -y --no-install-recommends google-cloud-sdk-gke-gcloud-auth-plugin

echo "🔥 Creating xpk cluster if it does not exist..."
xpk cluster create \
  --cluster="${CLUSTER}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --device-type="${DEVICE_TYPE}" \
  --num-slices=1 \
  --on-demand || echo "Cluster creation returned non-zero (might already exist), proceeding..."

echo "🔥 Launching xpk workload..."
xpk workload create \
  --cluster="${CLUSTER}" \
  --workload="${WORKLOAD_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --device-type="${DEVICE_TYPE}" \
  --base-docker-image="${IMAGE_TAG}" \
  --command="export JAX_COORDINATOR_ADDRESS=\"${WORKLOAD_NAME}-slice-job-0-0.${WORKLOAD_NAME}:1234\" && export JAX_PROCESS_COUNT=\"${MACHINE_COUNT}\" && export JAX_PROCESS_ID=\"\${JOB_COMPLETION_INDEX}\" && mkdir -p /gcs && gcsfuse --implicit-dirs /gcs && cd /workspace/orbax && PYTHONPATH=./checkpoint python3 -u -m orbax.checkpoint._src.testing.benchmarks.smart_batching --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --max_batch_size_mb=40960"
echo "✅ Success! Workload submitted: ${WORKLOAD_NAME}"
