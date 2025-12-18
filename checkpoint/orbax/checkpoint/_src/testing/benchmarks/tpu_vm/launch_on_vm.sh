#!/bin/bash
# Orchestrates Orbax benchmark execution on Cloud TPU VMs.

set -e

# --- Default Configuration ---
DEFAULT_ZONE="europe-west4-a"
DEFAULT_PROJECT="orbax-checkpoint"

# --- Global Variables ---
TPU_NAME=""
ZONE="$DEFAULT_ZONE"
PROJECT_ID="$DEFAULT_PROJECT"
# Default output bucket (overridable)
OUTPUT_DIR="gs://orbax-benchmarks/benchmark-results/${USER}"
CONFIG_FILE=""
REPO_URL="https://github.com/google/orbax.git"
BRANCH="main"
PR_NUMBER=""
JAX_VERSION="newest"
CUSTOM_COMMAND=""
SETUP_SCRIPT=""


# --- Helper Functions (Common) ---

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --tpu-name NAME      Name of the TPU VM / QR (default: $DEFAULT_TPU_NAME)"
    echo "  --zone ZONE          GCP Zone (default: $DEFAULT_ZONE)"
    echo "  --config FILE        Local path to benchmark config YAML (REQUIRED)"
    echo "  --output-dir GCS_URI GCS path for outputs (default: $DEFAULT_OUTPUT_DIR)"
    echo "  --project PROJECT    GCP Project ID (default: $DEFAULT_PROJECT)"
    echo "  --setup-script FILE  Path to setup script (optional). If provided, setup is run."
    echo "  --update             Run 'git pull' (fetch/checkout) on workers to update code."
    echo "  --repo-url URL       Git repo URL for setup (optional)"
    echo "  --branch BRANCH      Git branch/ref for setup (optional)"
    echo "  --pr NUMBER          GitHub PR number to setup/update (overrides --branch)"
    echo "  --jax-version VER    JAX version to install (newest, nightly, or X.Y.Z) (default: newest)"
    echo "  --command CMD        Custom command to run on all workers (overrides benchmark execution)"
    echo "                       Example: --command 'python3 -c \"import jax; print(jax.devices())\"'"
    echo "  --help               Show this help"
    exit 1
}

log() {
    echo ">>> $1" >&2
}

check_tpu_exists() {
    log "Verifying TPU resource existence..."
    # Check Queued Resources first (Multislice) then Standard TPU VM
    if ! gcloud alpha compute tpus queued-resources describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" >/dev/null 2>&1 && \
       ! gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
        echo "Error: TPU resource '$TPU_NAME' not found in zone $ZONE."
        echo "Please create it first using: ./manage_tpu.sh create ..."
        exit 1
    fi
}

# --- Standard Workflow (External) ---

run_standard() {
    log "Starting Standard Workflow (gcloud)..."

    # 1. Setup
    if [ -n "$SETUP_SCRIPT" ]; then
        log "Running Setup..."
        # SETUP_SCRIPT is already validated
        local setup_args="--repo-url \"$REPO_URL\" --branch \"$BRANCH\" --jax-version \"$JAX_VERSION\""
        if [ -n "$PR_NUMBER" ]; then setup_args="$setup_args --pr \"$PR_NUMBER\""; fi
        
        gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
          --worker=all --command "bash -s -- $setup_args" < "$SETUP_SCRIPT"
        log "Setup Complete."
    fi

    # 1b. Update Code (Git Fetch/Checkout)
    if [ "$DO_UPDATE" = true ]; then
        log "Updating Code (Git Fetch/Checkout)..."
        local git_cmd=""
        if [ -n "$PR_NUMBER" ]; then
             git_cmd="cd /app/orbax_repo && git fetch --depth 1 origin pull/$PR_NUMBER/head:pr_branch && git checkout pr_branch"
        else
             git_cmd="cd /app/orbax_repo && git fetch origin $BRANCH --depth=1 && git checkout FETCH_HEAD"
        fi
        
        gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
           --worker=all --command "$git_cmd"
        log "Update Complete."
    fi

    # 2. Upload Config
    if [ -n "$CONFIG_FILE" ]; then
        log "Uploading Config..."
        local remote_config="/tmp/orbax_config_$(date +%s).yaml"
        gcloud alpha compute tpus tpu-vm scp "$CONFIG_FILE" "$TPU_NAME:$remote_config" \
           --zone="$ZONE" --project="$PROJECT_ID" --worker=all
    fi

    # 3. Execution
    log "Launching Benchmark..."
    local run_cmd=$(build_run_command "$remote_config")
    
    gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
       --worker=all --command "$run_cmd"
    
    log "Benchmark Finished."
}


# --- Shared Utilities ---

build_run_command() {
    if [ -n "$CUSTOM_COMMAND" ]; then
        echo "$CUSTOM_COMMAND"
        return
    fi

    local remote_config=$1
    local cmd="export PYTHONPATH=$PYTHONPATH:/app/orbax_repo && \
               script=\$(find /app/orbax_repo -name run_benchmarks.py -print -quit) && \
               python3 \"\$script\" \
               --config_file=$remote_config \
               --alsologtostderr \
               --output_directory=$OUTPUT_DIR"
    echo "$cmd"
}

# --- Main Parsing ---

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tpu-name) TPU_NAME="$2"; shift ;;
        --zone) ZONE="$2"; shift ;;
        --project) PROJECT_ID="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --setup-script) SETUP_SCRIPT="$2"; shift ;;
        --update) DO_UPDATE=true ;;
        --repo-url) REPO_URL="$2"; shift ;;
        --branch) BRANCH="$2"; shift ;;
        --pr) PR_NUMBER="$2"; shift ;;
        --jax-version) JAX_VERSION="$2"; shift ;;
        --command) CUSTOM_COMMAND="$2"; shift ;;
        --discover-only) DISCOVER_ONLY=true ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done


if [ -z "$TPU_NAME" ]; then
    echo "Error: --tpu-name is required."
    usage
fi

if [ -z "$CONFIG_FILE" ] && [ -z "$CUSTOM_COMMAND" ] && [ -z "$SETUP_SCRIPT" ] && [ "$DO_UPDATE" != true ] && [ "$DISCOVER_ONLY" != true ]; then
    echo "Error: Must specify --config, --command, --setup-script, --update, or --discover-only."
    usage
fi

if [ -n "$SETUP_SCRIPT" ]; then
    if [ ! -f "$SETUP_SCRIPT" ]; then
        echo "Error: Setup script not found: $SETUP_SCRIPT"
        exit 1
    fi
fi

echo "=== Orbax TPU Launcher ==="
echo "Target: $TPU_NAME ($ZONE)"

# Verify resource availability before trying anything
check_tpu_exists

if [ "$DISCOVER_ONLY" = true ]; then
    log "Running Worker Discovery Only..."
    discover_all_workers
    exit 0
fi


run_standard
