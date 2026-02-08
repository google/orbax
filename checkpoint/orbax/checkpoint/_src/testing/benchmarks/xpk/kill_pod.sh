#!/bin/bash
set -e

# Script to delete a pod from a given workload after an optional delay.
#
# Prerequisites:
# - kubectl must be installed.
# - kubectl context must be configured to point to the correct GKE cluster.
#   You can configure this by running:
#   gcloud container clusters get-credentials <cluster_name> --zone <zone> --project <project_id>
#
# Example:
#   ./kill_pod.sh -w orbax-pyt-012813-a1b2c3 -d 120

usage() {
  echo "Usage: $0 -w <workload_name> [-d <delay_seconds>]"
  echo
  echo "Deletes one running pod associated with the specified workload after a delay."
  echo
  echo "  -w: Name of the XPK workload (mandatory)."
  echo "  -d: Delay in seconds before killing a pod (default: 60)."
  echo "  -h: Display this help message."
  echo
  echo "Example: $0 -w my-workload-abc -d 120"
  exit 1
}

DELAY=60
WORKLOAD_NAME=""

while getopts ":w:d:h" opt; do
  case ${opt} in
    w )
      WORKLOAD_NAME=$OPTARG
      ;;
    d )
      DELAY=$OPTARG
      ;;
    h )
      usage
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      usage
      ;;
    : )
      echo "Option -$OPTARG requires an argument." 1>&2
      usage
      ;;
  esac
done

if [[ -z "$WORKLOAD_NAME" ]]; then
  echo "Error: -w <workload_name> is mandatory."
  usage
fi

echo "Waiting for $DELAY seconds before searching for pods..."
sleep "$DELAY"

echo "Searching for running pods in workload: $WORKLOAD_NAME"
# Get names of pods that are in Running state and contain WORKLOAD_NAME in their name
POD_LIST=$(kubectl get pods --field-selector=status.phase=Running --no-headers -o custom-columns=":metadata.name" 2>/dev/null | grep "$WORKLOAD_NAME" || true)

if [[ -z "$POD_LIST" ]]; then
  echo "Error: No running pods found for workload '$WORKLOAD_NAME'."
  exit 1
fi

# Select one pod to delete (e.g., the first one in the list)
POD_TO_KILL=$(echo "$POD_LIST" | head -n 1)

echo "Found running pods:"
echo "$POD_LIST"
echo "Selected pod to kill: $POD_TO_KILL"

echo "Deleting pod $POD_TO_KILL..."
kubectl delete pod "$POD_TO_KILL"

echo "Pod $POD_TO_KILL deleted."
