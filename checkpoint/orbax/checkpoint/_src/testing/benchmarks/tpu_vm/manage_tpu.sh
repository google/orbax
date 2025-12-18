#!/bin/bash
# manage_tpu.sh
# Handles lifecycle of TPU VMs (Provisioning and Deletion).
# Supports standard Single Slice (TPU VM API) and Multislice (Queued Resources API).

set -e

# Default Configuration
DEFAULT_ZONE="europe-west4-a"
DEFAULT_PROJECT="orbax-checkpoint"
DEFAULT_NODE_COUNT=1

usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo "Commands:"
    echo "  create    Provision a new TPU VM or Queued Resource"
    echo "  delete    Delete the TPU VM or Queued Resource (auto-detects type)"
    echo "  status    Check the status of the TPU (auto-detects type)"
    echo "  list      List all TPU VMs and Queued Resources"
    echo "  versions  List available TPU runtime versions"
    echo ""
    echo "Options:"
    echo "  --tpu-name NAME      Name of the TPU VM / QR (will be prefixed with ${USER}-)"
    echo "                       Default suffix: $DEFAULT_TPU_SUFFIX"
    echo "  --zone ZONE          GCP Zone (default: $DEFAULT_ZONE)"
    echo "  --type TYPE          Accelerator type (default: $DEFAULT_TPU_TYPE)"
    echo "  --version VERSION    Runtime version (default: $DEFAULT_RUNTIME)"
    echo "  --node-count COUNT   Number of nodes for Multislice (default: 1)"
    echo "                       > 1 implies Queued Resource (CQR) mode."
    echo "  --spot               Use Spot VM (Preemptible/Cheaper)"
    echo "  --project PROJECT    GCP Project ID (default: $DEFAULT_PROJECT)"
    echo "  --help               Show this help"
    exit 1
}

# Parse Command
if [ $# -eq 0 ]; then
    usage
fi
COMMAND=$1
shift

# Parse Options
TPU_NAME=""
ZONE="$DEFAULT_ZONE"
TPU_TYPE=""
PROJECT_ID="$DEFAULT_PROJECT"
RUNTIME_VERSION=""
SPOT_FLAG=""
NODE_COUNT=$DEFAULT_NODE_COUNT

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tpu-name) TPU_NAME="$2"; shift ;;
        --zone) ZONE="$2"; shift ;;
        --type) TPU_TYPE="$2"; shift ;;
        --project) PROJECT_ID="$2"; shift ;;
        --version) RUNTIME_VERSION="$2"; shift ;;
        --node-count) NODE_COUNT="$2"; shift ;;
        --spot) SPOT_FLAG="--spot" ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Enforce Username Prefix
if [[ "$TPU_NAME" != "${USER}-"* ]]; then
    TPU_NAME="${USER}-${TPU_NAME}"
fi

# Determine Mode: Standard vs Multislice (QR)
IS_MULTISLICE=false
if [ "$NODE_COUNT" -gt 1 ]; then
    IS_MULTISLICE=true
fi

# Map --spot to --best-effort for Queued Resources
if [ "$IS_MULTISLICE" = true ] && [ "$SPOT_FLAG" == "--spot" ]; then
    SPOT_FLAG="--best-effort --provisioning-model=SPOT"
fi

# Set Base Commands
if [ "$IS_MULTISLICE" = true ]; then
    GCLOUD_BASE="gcloud alpha compute tpus queued-resources"
    # QR create has --node-count and uses --node-prefix or --node-id
    # We will use --node-id for QR name and --node-count for size
else
    GCLOUD_BASE="gcloud alpha compute tpus tpu-vm"
fi

COMMON_ARGS="--project=$PROJECT_ID --zone=$ZONE"

case $COMMAND in
    create)
        echo ">>> Creating TPU Resource '$TPU_NAME' ($TPU_TYPE) in $ZONE..."
        echo ">>> Runtime: $RUNTIME_VERSION"
        echo ">>> Node Count: $NODE_COUNT (Multislice: $IS_MULTISLICE)"
        
        if [ -z "$TPU_TYPE" ]; then
            echo "Error: --type is required for create command."
            exit 1
        fi
        
        if [ -z "$RUNTIME_VERSION" ]; then
            echo "Error: --version is required for create command."
            exit 1
        fi
        
        if [ "$IS_MULTISLICE" = true ]; then
            # Multislice (Queued Resource) Creation
             $GCLOUD_BASE create "$TPU_NAME" \
                $COMMON_ARGS \
                --accelerator-type "$TPU_TYPE" \
                --runtime-version "$RUNTIME_VERSION" \
                --node-count "$NODE_COUNT" \
                --node-prefix "$TPU_NAME" \
                --metadata "enable-oslogin=TRUE" \
                $SPOT_FLAG
        else
            # Standard Single Slice Creation
            $GCLOUD_BASE create "$TPU_NAME" \
                $COMMON_ARGS \
                --accelerator-type "$TPU_TYPE" \
                --version "$RUNTIME_VERSION" \
                --metadata "enable-oslogin=TRUE" \
                $SPOT_FLAG
        fi
            
        echo ">>> Create request submitted. Status:"
        $GCLOUD_BASE describe "$TPU_NAME" $COMMON_ARGS --format="value(state)"
        ;;
        
    delete)
        echo ">>> Deleting TPU Resource '$TPU_NAME'..."
        # Try Queued Resource first (Multislice)
        if gcloud alpha compute tpus queued-resources describe "$TPU_NAME" $COMMON_ARGS >/dev/null 2>&1; then
            echo ">>> Identified as Queued Resource (Multislice). Deleting..."
            gcloud alpha compute tpus queued-resources delete "$TPU_NAME" $COMMON_ARGS --force --quiet
        # Try Standard TPU VM second
        elif gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" $COMMON_ARGS >/dev/null 2>&1; then
            echo ">>> Identified as Standard TPU VM. Deleting..."
            gcloud alpha compute tpus tpu-vm delete "$TPU_NAME" $COMMON_ARGS --quiet
        else
            echo "Error: Resource '$TPU_NAME' not found (checked both Queued Resources and TPU VMs)."
            exit 1
        fi
        echo ">>> Delete request submitted."
        ;;
        
    status)
        echo ">>> Status of '$TPU_NAME'..."
        RESULT_FOUND=false
        
        # Check Queued Resource
        if QR_STATE=$(gcloud alpha compute tpus queued-resources describe "$TPU_NAME" $COMMON_ARGS --format="value(state.state)" 2>/dev/null); then
             echo ">>> Identified as Queued Resource (Multislice)"
             echo "State: $QR_STATE"
             STATE=$QR_STATE
             RESULT_FOUND=true
        # Check Standard TPU VM
        elif VM_STATE=$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" $COMMON_ARGS --format="value(state)" 2>/dev/null); then
             echo ">>> Identified as Standard TPU VM"
             echo "State: $VM_STATE"
             STATE=$VM_STATE
             RESULT_FOUND=true
        else
             echo "State: NOT_FOUND"
        fi
        
        if [ "$RESULT_FOUND" = true ] && [[ "$STATE" == "READY" || "$STATE" == "ACTIVE" ]]; then
             echo ""
             echo "Resource is Ready! You can now run benchmarks:"
             echo ""
             echo "  # Launch benchmark setup & run"
             echo "  ./launch_on_vm.sh --setup-script ./setup_tpu.sh --config ../configs/pytree_checkpoint_benchmark.yaml --tpu-name $TPU_NAME --zone $ZONE"
             echo ""
        fi
        ;;
        
    versions)
        echo ">>> Available TPU Runtime Versions in $ZONE:"
        gcloud compute tpus versions list --zone "$ZONE" --project "$PROJECT_ID"
        ;;

    list)
        echo ">>> Listing Single Slice TPU VMs in $ZONE:"
        gcloud alpha compute tpus tpu-vm list --zone "$ZONE" --project "$PROJECT_ID"
        echo ""
        echo ">>> Listing Multislice Queued Resources in $ZONE:"
        gcloud alpha compute tpus queued-resources list --zone "$ZONE" --project "$PROJECT_ID"
        ;;
        
    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
