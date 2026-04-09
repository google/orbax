#!/bin/bash
# Runs Llama 70B benchmarks across various slice counts and replica parallel settings.
# For each run, a temporary YAML config is created by dynamically setting the
# 'use_replica_parallel' option based on the loop variable.

# Defaults
NUM_SLICES=(2 4 8 16 32)
REPLICA_PARALLEL_VALUES=(true false)
CLUSTER_NAME=""
ZONE=""
PROJECT="orbax-checkpoint"
DOCKER_IMAGE="gcr.io/orbax-checkpoint/orbax-benchmarks:main"
MAX_RESTARTS=20
NUM_SAVINGS=20

function print_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --cluster_name CLUSTER  GKE cluster name (REQUIRED)"
  echo "  --zone ZONE            GCP zone (REQUIRED)"
  echo "  --num_slices \"S1 S2\"   Space-separated list of slice counts (Default: ${NUM_SLICES[*]})"
  echo "  --project PROJECT      GCP project (Default: $PROJECT)"
  echo "  --docker_image IMAGE   Benchmark docker image (Default: $DOCKER_IMAGE)"
  echo "  --max_restarts N       Maximum job restarts (Default: $MAX_RESTARTS)"
  echo "  --num_savings N        Number of savings to run (Default: $NUM_SAVINGS)"
  echo "  --help, -h             Print this help message"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --cluster_name)
      CLUSTER_NAME="$2"
      shift 2
      ;;
    --num_slices)
      IFS=' ' read -r -a NUM_SLICES <<< "$2"
      shift 2
      ;;
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --docker_image)
      DOCKER_IMAGE="$2"
      shift 2
      ;;
    --max_restarts)
      MAX_RESTARTS="$2"
      shift 2
      ;;
    --num_savings)
      NUM_SAVINGS="$2"
      shift 2
      ;;
    --help | -h)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

if [[ -z "$CLUSTER_NAME" ]]; then
  echo "Error: --cluster_name is required."
  echo "Use --help for usage information."
  exit 1
fi

if [[ -z "$ZONE" ]]; then
  echo "Error: --zone is required."
  echo "Use --help for usage information."
  exit 1
fi

CONFIG_DIR="orbax/checkpoint/_src/testing/benchmarks/configs/evaluations/replica_parallel/cpu"

cleanup() {
  echo "Cleaning up temporary config files..."
  rm -f ${CONFIG_DIR}/llama-70b-replicas-*-temp-*.yaml
}
trap cleanup EXIT

for num_slices in "${NUM_SLICES[@]}"; do
  for replica_parallel in "${REPLICA_PARALLEL_VALUES[@]}"; do
    SOURCE_CONFIG="${CONFIG_DIR}/llama-70b-replicas-$num_slices.yaml"
    TEMP_CONFIG="${CONFIG_DIR}/llama-70b-replicas-$num_slices-temp-$replica_parallel.yaml"

    echo "Generating $TEMP_CONFIG from $SOURCE_CONFIG"
    # Create temp config with specific values, replacing any existing value.
    sed -e "s/use_replica_parallel:.*/use_replica_parallel: [$replica_parallel]/" \
        -e "s/num_savings:.*/num_savings: $NUM_SAVINGS/" \
        "$SOURCE_CONFIG" > "$TEMP_CONFIG"

    python3 orbax/checkpoint/_src/testing/benchmarks/xpk/launch_xpk.py \
      --project "$PROJECT" \
      --zone "$ZONE" \
      --cluster_name "$CLUSTER_NAME" \
      --device_type n2-standard-4-64 \
      --num_slices "$num_slices" \
      --config_file "$TEMP_CONFIG" \
      --docker_image "$DOCKER_IMAGE" \
      --output_directory "gs://orbax-benchmarks/${USER}/llama70b-cpu-replicas-$num_slices-rp-$replica_parallel/" \
      --nodelete_cluster_on_completion \
      --nocreate_cluster \
      --max_restarts "$MAX_RESTARTS"
  done
done
