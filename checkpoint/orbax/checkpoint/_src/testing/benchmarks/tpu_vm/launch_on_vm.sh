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
RAMFS_DIR=""


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
    echo "  --test_restart_workflow If True, run workload creation and execution twice to test restart."
    echo "  --repo-url URL       Git repo URL for setup (optional)"
    echo "  --branch BRANCH      Git branch/ref for setup (optional)"
    echo "  --pr NUMBER          GitHub PR number to setup/update (overrides --branch)"
    echo "  --jax-version VER    JAX version to install (newest, nightly, or X.Y.Z) (default: newest)"
    echo "  --command CMD        Custom command to run on all workers (overrides benchmark execution)"
    echo "                       Example: --command 'python3 -c \"import jax; print(jax.devices())\"'"
    echo "  --ramfs-dir DIR      Path to ramfs directory for setup (optional)"
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

# --- Standard Workflow ---

# Discover all TPU nodes (slices)
discover_nodes() {
    local nodes=()
    
    # Check for Queued Resource
    if gcloud alpha compute tpus queued-resources describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log "Detected Queued Resource: $TPU_NAME"
        # Discovery via list filter
        local nodes_list=$(gcloud alpha compute tpus tpu-vm list --zone="$ZONE" --project="$PROJECT_ID" --filter="name~${TPU_NAME}*" --format="value(name)")
        if [ -n "$nodes_list" ]; then
            mapfile -t nodes <<< "$nodes_list"
            log "QR Node Count: ${#nodes[@]}"
        else
            echo "Error: No nodes found matching ${TPU_NAME}*. Please verify TPU name." >&2
            exit 1
        fi
    else
        log "Assuming Standard TPU VM: $TPU_NAME"
        nodes+=("$TPU_NAME")
    fi
    # Print nodes array space-separated
    echo "${nodes[@]}"
}

# Discover workers for a given node
# Returns a list of hostnames (one per line)
get_workers_for_node() {
    local node=$1
    log "Scanning node: $node"
    
    # Try efficient JSON path first
    local json_filter="value(query_value.items.filter(key:hostname).value)"
    local raw_workers=$(gcloud alpha compute tpus tpu-vm get-guest-attributes "$node" \
        --zone="$ZONE" --project="$PROJECT_ID" --format="$json_filter" 2>/dev/null || true)

    # Fallback to grep if specific JMESPath fails
    if [ -z "$raw_workers" ]; then
         raw_workers=$(gcloud alpha compute tpus tpu-vm get-guest-attributes "$node" \
            --zone="$ZONE" --project="$PROJECT_ID" --format="json" 2>/dev/null \
            | grep -A 2 '"key": "hostname"' | grep '"value":' | awk -F'"' '{print $4}')
    fi
    
    echo "$raw_workers"
}

# Discover all worker hostnames across all nodes (Internal only)
discover_all_workers() {
    # Capture output into array
    read -r -a nodes <<< "$(discover_nodes)"
    
    # Iterate and collect workers
    local all_workers=()
    for node in "${nodes[@]}"; do
        local node_workers=$(get_workers_for_node "$node")
        if [ -n "$node_workers" ]; then
            while IFS= read -r worker; do
                all_workers+=("$worker")
            done <<< "$node_workers"
        else
            log "Warning: No workers found for node $node."
        fi
    done
    
    if [ ${#all_workers[@]} -eq 0 ]; then
        echo "Error: No workers found across any nodes." >&2
        exit 1
    fi
    
    echo "${all_workers[@]}"
}

run_standard() {
    log "Starting Standard Workflow (gcloud)..."

    # Discover nodes (handles Multislice/QR)
    read -r -a nodes <<< "$(discover_nodes)"
    log "Target Nodes (${#nodes[@]}): ${nodes[@]}"

    # Shared config path
    local remote_config=""
    if [ -n "$CONFIG_FILE" ]; then
        remote_config="/tmp/orbax_config_$(date +%s).yaml"
    fi

    # Phase 0: Sync SSH Keys (Sequential)
    # Prevents "Multiple concurrent mutations" error by ensuring keys are pushed once.
    if [ ${#nodes[@]} -gt 0 ]; then
        log "Phase 0: Synchronizing SSH keys (Sequential)..."
        # Dry-run connection to first node to trigger key propagation
        gcloud alpha compute tpus tpu-vm ssh "${nodes[0]}" --zone="$ZONE" --project="$PROJECT_ID" \
            --worker=0 --command "true" || true
    fi

    # Phase 1: Setup, Update, Upload (Parallel)
    log "Phase 1: Setup and Configuration..."
    local setup_pids=()
    for node in "${nodes[@]}"; do
        (
            log "Configuring node: $node"
            # 1. Setup
            if [ -n "$SETUP_SCRIPT" ]; then
                local setup_args="--repo-url \"$REPO_URL\" --branch \"$BRANCH\" --jax-version \"$JAX_VERSION\""
                if [ -n "$PR_NUMBER" ]; then setup_args="$setup_args --pr \"$PR_NUMBER\""; fi
                if [ -n "$RAMFS_DIR" ]; then setup_args="$setup_args --ramfs-dir \"$RAMFS_DIR\""; fi
                
                gcloud alpha compute tpus tpu-vm ssh "$node" --zone="$ZONE" --project="$PROJECT_ID" \
                  --worker=all --command "bash -s -- $setup_args" < "$SETUP_SCRIPT"
            fi

            # 1b. Update Code
            if [ "$DO_UPDATE" = true ]; then
                local git_cmd=""
                if [ -n "$PR_NUMBER" ]; then
                     git_cmd="cd /app/orbax_repo && git fetch --depth 1 origin pull/$PR_NUMBER/head:pr_branch && git checkout pr_branch"
                else
                     git_cmd="cd /app/orbax_repo && git fetch origin $BRANCH --depth=1 && git checkout FETCH_HEAD"
                fi
                
                gcloud alpha compute tpus tpu-vm ssh "$node" --zone="$ZONE" --project="$PROJECT_ID" \
                   --worker=all --command "$git_cmd"
            fi

            # 2. Upload Config
            if [ -n "$CONFIG_FILE" ]; then
                gcloud alpha compute tpus tpu-vm scp "$CONFIG_FILE" "$node:$remote_config" \
                   --zone="$ZONE" --project="$PROJECT_ID" --worker=all
            fi
        ) &
        setup_pids+=($!)
    done
    
    # Wait for setup to fail/complete
    for pid in "${setup_pids[@]}"; do
        wait "$pid"
        if [ $? -ne 0 ]; then
             echo "Error: Setup failed on one or more nodes." >&2
             exit 1
        fi
    done
    log "Phase 1 Complete."

    # Phase 2: Execution (Parallel)
    log "Phase 2: Execution..."
    local run_cmd=$(build_run_command "$remote_config")
    local exec_pids=()
    
    for node in "${nodes[@]}"; do
        (
            gcloud alpha compute tpus tpu-vm ssh "$node" --zone="$ZONE" --project="$PROJECT_ID" \
               --worker=all --command "$run_cmd"
        ) &
        exec_pids+=($!)
    done
    
    # Wait for execution
    local failures=0
    for pid in "${exec_pids[@]}"; do
        wait "$pid" || failures=$((failures+1))
    done
    
    if [ "$failures" -ne 0 ]; then
        echo "Error: Execution failed on $failures node(s)." >&2
        exit 1
    fi
    
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
    if [ -n "$RAMFS_DIR" ]; then
        cmd="$cmd --local_directory=$RAMFS_DIR"
    fi
    echo "$cmd"
}

build_cmd() {
    local cmd="
    # sudo rm -f /tmp/libtpu_lockfile
    cd /app/orbax_repo
    export PYTHONPATH=\$PYTHONPATH:.

    script=\$(find /app/orbax_repo -name run_tests.py -print -quit)
    if [ -z \"\$script\" ]; then
        echo \"Error: run_tests.py not found in /app/orbax_repo\"
        exit 1
    fi

    export JAX_TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
    export JAX_DISTRIBUTED_SHUTDOWN_TIMEOUT=10
    export ALLOW_MULTIPLE_LIBTPU_LOAD=1

    (
        export JAX_PLATFORMS=tpu
        export TPU_VISIBLE_DEVICES=0,1,2,3
        export JAX_PROCESS_ID=0
        export JAX_NUM_PROCESSES=2
        export JAX_DISTRIBUTED_SERVICE_ADDR=localhost:1234
        export JAX_NUM_TASKS=2
        export JAX_TASK_ID=0
        export NUM_PROCESSES=2
        export MULTIPROCESS_TEST_WORKER_ID=0
        export JAX_ALLOW_UNUSED_TPUS=true
        export JAX_PORT=1234
        python3 -c \"import jax; jax.distributed.initialize(coordinator_address='localhost:1234', num_processes=2, process_id=0, local_device_ids=[0,1,2,3]); print(jax.devices());\"
    ) &

    sleep 5

    (
        export JAX_PLATFORMS=tpu
        export TPU_VISIBLE_DEVICES=4,5,6,7
        export JAX_ALLOW_UNUSED_TPUS=true
        export JAX_PROCESS_ID=1
        export JAX_NUM_PROCESSES=2
        export JAX_DISTRIBUTED_SERVICE_ADDR=localhost:1234
        export JAX_NUM_TASKS=2
        export JAX_TASK_ID=1
        export NUM_PROCESSES=2
        export MULTIPROCESS_TEST_WORKER_ID=1
        python3 -c \"import jax; jax.distributed.initialize(coordinator_address='localhost:1234', num_processes=2, process_id=1, local_device_ids=[4,5,6,7]); print(jax.devices());\"
    )

    wait
    "
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
        --test_restart_workflow) TEST_RESTART_WORKFLOW=true ;;
        --repo-url) REPO_URL="$2"; shift ;;
        --branch) BRANCH="$2"; shift ;;
        --pr) PR_NUMBER="$2"; shift ;;
        --jax-version) JAX_VERSION="$2"; shift ;;
        --command) CUSTOM_COMMAND="$2"; shift ;;
        --ramfs-dir) RAMFS_DIR="$2"; shift ;;
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

if [ -z "$CONFIG_FILE" ] && [ -z "$CUSTOM_COMMAND" ] && [ -z "$SETUP_SCRIPT" ] && [ "$DO_UPDATE" != true ] && [ "$TEST_RESTART_WORKFLOW" != true ] && [ "$DISCOVER_ONLY" != true ]; then
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
