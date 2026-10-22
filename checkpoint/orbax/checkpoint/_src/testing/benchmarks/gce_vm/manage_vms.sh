#!/bin/bash
# Creates and deletes the GCE CPU VMs that gce_vm/launch.sh runs SimDevice
# benchmarks on: NAME-0 ... NAME-{COUNT-1}, on Container-Optimized OS, which
# comes with Docker.

set -e

DEFAULT_ZONE="europe-west4-a"
DEFAULT_PROJECT="orbax-checkpoint"
DEFAULT_MACHINE_TYPE="n2-standard-32"

usage() {
  echo "Usage: $0 create|delete|list [OPTIONS]"
  echo "Options:"
  echo "  --name NAME            VM name prefix (default: \${USER}-simdevice)"
  echo "  --count N              Number of VMs (create/delete; default: 2)"
  echo "  --machine-type TYPE    Machine type (default: $DEFAULT_MACHINE_TYPE)"
  echo "  --zone ZONE            GCP zone (default: $DEFAULT_ZONE)"
  echo "  --project PROJECT      GCP project (default: $DEFAULT_PROJECT)"
  echo "  --spot                 Use Spot VMs"
  echo "  --help                 Show this help"
  exit 1
}

[[ $# -eq 0 ]] && usage
COMMAND=$1
shift

NAME="${USER}-simdevice"
COUNT=2
MACHINE_TYPE="$DEFAULT_MACHINE_TYPE"
ZONE="$DEFAULT_ZONE"
PROJECT_ID="$DEFAULT_PROJECT"
declare -a SPOT_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --name) NAME="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT_ID="$2"; shift 2 ;;
    --spot) SPOT_ARGS=(--provisioning-model=SPOT); shift 1 ;;
    --help) usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

declare -a VMS=()
for ((i = 0; i < COUNT; i++)); do
  VMS+=("${NAME}-${i}")
done

case $COMMAND in
  create)
    # cloud-platform scope: the VMs pull the image from gcr.io and write
    # results to GCS with their service account.
    gcloud compute instances create "${VMS[@]}" \
      --project="$PROJECT_ID" \
      --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
      --image-family=cos-stable \
      --image-project=cos-cloud \
      --boot-disk-size=100GB \
      --scopes=cloud-platform \
      "${SPOT_ARGS[@]}"
    ;;
  delete)
    gcloud compute instances delete "${VMS[@]}" \
      --project="$PROJECT_ID" --zone="$ZONE" --quiet
    ;;
  list)
    gcloud compute instances list \
      --project="$PROJECT_ID" \
      --filter="name ~ ^${NAME}-[0-9]+$ AND zone:${ZONE}" \
      --format="table(name, machineType.basename(), status, networkInterfaces[0].networkIP)"
    ;;
  *)
    usage
    ;;
esac
