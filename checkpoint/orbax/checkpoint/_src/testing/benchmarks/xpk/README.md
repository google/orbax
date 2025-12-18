# Orbax Checkpoint XPK Benchmarks


**The Production Tooling for Orbax Benchmarks on Cloud TPUs.**

This directory provides a push-button experience for running distributed JAX
benchmarks on Google Kubernetes Engine (GKE) via XPK.

---

## ⚡ TL;DR: Quick Start

**1. Build the Image** (Only when code changes)

```bash
./build_image.sh --project=orbax-checkpoint --tag=stable-v1
```

**2. Run the Benchmark** (Iterate on configs freely)

```bash
python3 launch_xpk.py \
  --cluster_name=orbax-dev-cluster \
  --tpu_type=v5litepod-8 \
  --zone=us-west1-c \
  --config_file=../configs/pytree_checkpoint_benchmark.yaml \
  --docker_image=gcr.io/orbax-checkpoint/orbax-benchmarks:stable-v1 \
  --output_directory=gs://orbax-benchmarks/runs/$(date +%Y%m%d)
```

---

## 📚 Table of Contents
1.  [Concept & Architecture](#1-concept-architecture)
2.  [Prerequisites](#2-prerequisites)
3.  [Workflow Part 1: The Build](#3-workflow-part-1-the-build-docker)
4.  [Workflow Part 2: The Run](#4-workflow-part-2-the-run-launch_xpkpy) (Inner Loop)
5.  [Internals: Inside the Container](#5-internals-inside-the-container-run_benchmarkspy)
6.  [Troubleshooting & Debugging](#6-troubleshooting--debugging)
7.  [Cleanup](#7-cleanup-save-money) (**IMPORTANT**)

---

## 1. Concept & Architecture {#1-concept-architecture}

We use a **Controller-Worker** model. The `launch_xpk` script (Controller)
provisions resources and schedules the workload. The GKE Pods (Workers) pull
your image and execute the actual benchmark code using `jax.distributed`.

```sequence-diagram
participant Local
participant GCR
participant GCS
participant XPK
participant TPU

Local->GCR: 1. Build & Push Image
Local->GCS: 2. Upload Config (.yaml)
Local->XPK: 3. Request Cluster
XPK-->Local: Cluster Ready
Local->XPK: 4. Submit Workload
XPK->TPU: 5. Schedule Pods
TPU->GCR: 6. Pull Image
TPU->GCS: 7. Download Config
TPU->TPU: 8. Run Benchmark
TPU->GCS: 9. Upload Metrics
Local->XPK: 10. Poll Status & Logs
```

### 📂 Directory Structure

*   `xpk/`: This directory. Contains orchestration scripts and Dockerfile.
*   `../configs/`: YAML configuration files defining benchmark parameters (Top-level configs).
*   `../run_benchmarks.py`: The actual Python entrypoint running inside the container.

---

## 2. Prerequisites {#2-prerequisites}

### GCP Project & Authentication
Ensure you have a GCP project with TPU quota (e.g., `orbax-checkpoint`).

```bash
# 1. Login to GCP User Account
gcloud auth login

# 2. Login Application Default Credentials (ADC) - Critical for logic inside scripts
gcloud auth application-default login

# 3. Configure Docker to use gcloud credentials (required for pushing images)
gcloud auth configure-docker
```

### Tool Installation

*   **XPK**: [Installation Guide](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md)
*   **Docker**: [Get Docker](https://docs.docker.com/get-docker/)

### 🔗 Useful Resources

*   **[TPU Accelerator Zones](https://docs.cloud.google.com/compute/docs/regions-zones/accelerator-zones)**


---

## 3. Workflow Part 1: The Build (Docker) {#3-workflow-part-1-the-build-docker}

**Goal**: Package the Orbax source code and dependencies into a reproducible
artifact.

### 📦 What is inside the Image? (`Dockerfile` Deep Dive)
The `Dockerfile` generates a lean, production-ready environment optimized for
JAX on TPU.

*   **Base OS**: `python:3.11-slim` (Debian-based, minimized footprint).
*   **System Tools**:
    *   `dnsutils`: **Critical** for `jax.distributed` to resolve Pod hostnames in GKE.
    *   `git`: Required for fetching Orbax source code.
*   **Python Libraries**:
    *   `jax[tpu]`, `jaxlib`: Configurable versions (Stable/Nightly).
    *   `tensorflow`: Required for `tensorboard` summary writing.
    *   `gcsfs`: For direct GCS I/O.
    *   `portpicker`: For finding free ports in distributed tests.
    *   `clu`: Google's Common Loop Utils for logging/metrics.
*   **Target Code (`orbax`)**: Use specific PRs or Branches. The image does *not* copy your local directory; it fetches fresh code from GitHub.

### 🛠️ Common Build Commands

**Scenario A: Production / Stable Build**

```bash
# Builds from 'main' branch, installs stable JAX
./build_image.sh --project=orbax-checkpoint --tag=stable-v1
```

**Scenario B: Testing a Pull Request**

```bash
# Fetches code from PR #1234, helpful for testing pre-submit changes on TPU
./build_image.sh --pr=1234 --tag=pr-1234
```

**Scenario C: Debugging JAX Regressions**

```bash
# Installs JAX Nightly to test latest compiler features
./build_image.sh --jax-version=nightly --tag=jax-nightly-debug
```


<!-- LINT.IfChange(build_image_flags_table) -->
### Flag Reference (`build_image.sh`)

| Flag | Default | Description | When to use? |
| :--- | :--- | :--- | :--- |
| `--project` | `gcloud config` | GCP Project ID for the Container Registry. | **Always**, unless your default gcloud config is correct. |
| `--tag` | *(Auto)* | Image tag (e.g., `v1`, `pr-123`). | **Always**. Use unique tags to prevent GKE caching stale code. |
| `--pr` | `None` | GitHub PR Number (e.g., `123`). | **CI/CD**. When verifying a specific PR before merging. |
| `--branch` | `main` | GitHub Branch name. | **Feature Dev**. When building from a feature branch. |
| `--jax-version` | `newest` | `newest`, `nightly`, or `x.y.z`. | **Debugging**. Use `nightly` to test bleeding-edge JAX features. |
| `--device` | `tpu` | `tpu`, `gpu`, `cpu`. | **Multi-Hardware**. When testing on GPU or CP/Local validation. |
| `--base-image` | `python:3.11...` | Base Docker Image. | **Advanced**. If you need custom drivers or non-standard OS libs. |

---
<!-- LINT.ThenChange(build_image.sh:build_image_flags) -->

## 4. Workflow Part 2: The Run (`launch_xpk.py`) {#4-workflow-part-2-the-run-launch_xpkpy}

**Goal**: Provision a TPU cluster (if needed) and run the workload.

This script is the **Control Plane**. It wraps `xpk workoad create` and handles
the complexities of cluster management and config propagation.

### 🔄 The "Inner Loop" (Development Workflow)

> IMPORTANT:
> **Docker Caching in GKE (The "It didn't change!" Trap)**
> GKE nodes **aggressively cache** Docker images by tag.
>
> *   If you push `orbax-benchmarks:dev`, run a job, change code, push
>     `orbax-benchmarks:dev` again, and run another job...
> *   **GKE will likely use the OLD image** if the node already has it cached.
>     It often ignores that you overwrote the tag in GCR.
>
> **Best Practice**: ALWAYS use unique tags for every code change. The
> `build_image.sh` script does this automatically (e.g., `pr-123-tpu-nightly`)
> or you can pass a custom one.

#### Decision Tree: Do I need to Rebuild?

| Change Type | Action Required | Why? |
| :--- | :--- | :--- |
| **Python Code** | **MUST Build** & **Push** | Source code is baked into `/app/orbax_repo`. K8s cannot see your local edits. |
| **Dependencies** | **MUST Build** & **Push** | Installed in the image layer. |
| **YAML Config** | **NO Build** | `launch_xpk.py` uploads your local config file directly to the container at runtime. |
| **JAX Flags** | **NO Build** | Can often be set via `XLA_FLAGS` env var in `launch_xpk.py`. |

#### Recommended Dev Flow
1.  **Edit Code** (Locally).
2.  **Build & Tag**: `./build_image.sh --tag=dev-$(date +%s) ...` (Using timestamp ensures uniqueness).
3.  **Launch**: `python3 launch_xpk.py --docker_image=...:dev-<timestamp> ...`

<!-- LINT.IfChange(launch_xpk_flags_table) -->
### 🎛️ Key Flags

#### Cluster Hardware

*   `--cluster_name`: (Required) ID for your cluster.
*   `--tpu_type`: e.g., `v5litepod-8`, `v4-8`. Required if creating a cluster.
*   `--zone`: e.g., `us-west1-c`.
*   `--create_cluster`: (Default: `True`) Auto-provisions if missing.
*   `--delete_cluster_on_completion`: (Default: `True`) **Set to False for debugging** or if you want to reuse the cluster for sequential runs.
*   `--spot`: (Default: `True`) Uses Spot VMs (Preemptible). Much cheaper but can be interrupted.
*   `--on_demand`: Uses On-Demand VMs. More expensive, no interruptions.
*   `--custom_cluster_arguments`: Pass raw arguments to the underlying GKE creation (e.g., `--network=my-net`).
*   `--private`: Create a private GKE cluster (no public IPs on nodes).
*   `--authorized_networks`: CIDR ranges allowed to access the control plane in private clusters.

### 🎛️ Flag Reference (`launch_xpk.py`)

#### 🛠️ Cluster Configuration
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--cluster_name` | *Required* | Name of the XPK/GKE cluster. |
| `--tpu_type` | `None` | TPU accelerator type (e.g., `v5litepod-8`, `v4-8`). Required when creating a cluster. |
| `--num_slices` | `1` | Number of slices (Multislice). |
| `--zone` | `europe-west4-a` | GCP Zone. |
| `--create_cluster` | `True` | Auto-create cluster if it doesn't exist. |
| `--delete_cluster_on_completion` | `True` | **Important**: Auto-delete cluster after workload finishes. Set to `False` to keep it for debugging. |
| `--custom_cluster_arguments` | `None` | Pass raw args to GKE creation (e.g., `--network=my-vpc`). |
| `--enable_tpu_autolock` | `False` | Enable TPU autolock feature. |
| `--project` | `orbax-checkpoint` | GCP Project ID. |
| `--cpu_limit` | `None` | (Advanced) CPU limit for the cluster (e.g., `112`). |
| `--memory_limit` | `None` | (Advanced) Memory limit for the cluster (e.g., `192Gi`). |
| `--device_type` | `None` | **GPU Support**: Specify device type (e.g., `h100-mega-80gb-8`). |

#### 💰 Capacity & Scheduling
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--spot` | `True` | Use Spot VMs (cheaper, preemptible). |
| `--on_demand` | `False` | Use On-Demand VMs (expensive, stable). |
| `--reservation` | `None` | Use a specific GCP Reservation ID. |
| `--priority` | `medium` | Workload priority: `very-low`, `low`, `medium`, `high`, `very-high`. |
| `--scheduler` | `None` | Explicitly set scheduler (e.g., `kueue`). |
| `--max_restarts` | `0` | Auto-restart workload N times on failure. |

#### 📦 Workload & Environment
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--config_file` | *Required* | Path to local YAML benchmark config. |
| `--output_directory` | *Required* | GCS path (`gs://bucket/path`) for results/artifacts. |
| `--docker_image` | `...:stable` | Docker image to run. |
| `--env` | `[]` | List of env vars: `--env=KEY=VAL`. |
| `--env_file` | `None` | Path to file with env vars. |
| `--ramdisk_directory` | `None` | Mount a ramdisk at this path (e.g., `/tmp/ramdisk`). |
| `--sa` | `None` | Kubernetes Service Account to run as. |

#### 🔌 Networking & Security
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--private` | `False` | Create a private GKE cluster (no public node IPs). |
| `--authorized_networks` | `[]` | CIDR ranges allowed to access control plane (e.g., `10.0.0.0/8`). |

#### 🔍 Debugging & Observability
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--verbose` | `False` | Print all underlying XPK/Shell commands. |
| `--skip_preflight_checks` | `False` | Skip Docker/GCS/Gcloud checks (faster). |
| `--restart_on_user_code_failure` | `False` | **Critical**: If True, K8s restarts job on app crash. Default `False` helps debug crashes. |
| `--enable_ops_agent` | `False` | Install Google Cloud Ops Agent for system metrics. |
| `--debug_dump_gcs` | `None` | GCS path to dump XLA debug artifacts. |

#### 🔁 Lifecycle & Automation
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--wait` | `True` | Wait for workload completion. If `False`, script exits immediately after launch. |
| `--workload_name` | *(Auto-generated)* | Custom name for the workload. |
| `--run_name` | `None` | Overrides the generated run name. |
| `--docker_name` | `None` | Name of the Docker container. |

#### 🛣️ Pathways Mode (Advanced)
*Pathways uses a "Sidecar" architecture: Client (Your Code) <-> Proxy <->
Server (TPU Manager).*

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--enable_pathways` | `False` | Enable Pathways backend (Single-Controller). |
| `--pathways_server_image` | `...:latest` | Pathways server image (manages TPU chips). |
| `--pathways_proxy_image` | `...:latest` | Pathways proxy image (bridges user code to server). |
<!-- LINT.ThenChange(launch_xpk.py:launch_xpk_flags) -->


---

## 5. Internals: Inside the Container (`run_benchmarks.py`) {#5-internals-inside-the-container-run_benchmarkspy}
Once `launch_xpk.py` schedules the job, Kubernetes pulls your Docker image and
starts the entrypoint: `run_benchmarks.py`.

### 🧠 Architecture & Design
1.  **Config Injection (Not Baking)**: We do **NOT** bake configs into the
    Docker image. `launch_xpk.py` uploads your local YAML to GCS, and the
    container downloads it at runtime. This allows 3-second iteration loops on
    hyperparameters without rebuilding Docker.
2.  **Distributed Init**: The script automatically calls
    `jax.distributed.initialize()`. It relies on XPK/Kueue setting
    `JAX_COORDINATOR_ADDRESS`, `JAX_PROCESS_ID`, etc. If running locally (single
    process), it gracefully skips this.
3.  **HLO Dumping**: If `--debug_dump_gcs` is passed to the launcher, it injects `XLA_FLAGS=--xla_dump_to=...` into the container environment, streaming HLO protos to GCS for compiler debugging.

### Flag Reference (`run_benchmarks.py`)
*Usually set automatically by `launch_xpk.py`, but good to know for debugging.*

| Flag | Description |
| :--- | :--- |
| `--config_file` | Local path to the YAML config (inside the container). |
| `--output_directory` | GCS path for results. |
| `--enable_hlo_dump` | If set, logs HLO protos. |

---

## 6. Troubleshooting & Debugging {#6-troubleshooting--debugging}

### 🕵️‍♂️ Debugging Matrix

| Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| **`ImagePullBackOff`** | Typo or Permission | Check `gcloud auth configure-docker`. Verify `--project` matches the registry. |
| **`Pending` (Forever)** | No capacity / Quota | Switch to `--on_demand` or check detailed quota. Use `kubectl get events`. |
| **CrashLoopBackOff** | Code Bug | Run with `--delete_cluster_on_completion=False`. Log in with `kubectl logs`. |
| **Hang at Init** | DNS / Network | `dnsutils` missing? Check `run_benchmarks.py` logs for distributed init failures. |
| **"No accessible resources"** | Priority | Try `--priority=medium` or `--priority=high` (if on internal queue). |

### 🛠️ Manual Debugging (The "Break Glass" method)

**1. Get Cluster Credentials**
If you want to use `kubectl` directly:

```bash
gcloud container clusters get-credentials <CLUSTER_NAME> --zone <ZONE> --project <PROJECT_ID>
```

**2. List Workloads**

```bash
xpk workload list --cluster <CLUSTER_NAME>
```

**3. Stream Logs**

```bash
xpk inspector --cluster <CLUSTER_NAME> --workload <WORKLOAD_ID>
```

**4. Visual Debugging (GCP Console)**
> **[GCP Kubernetes Engine Dashboard](https://console.cloud.google.com/kubernetes/list)**
>
> 1.  Go to **Workloads**.
> 2.  Find your `job-xxx`.
> 3.  Click **Logs** tab. You can filter by `severity=ERROR`.

---

## 7. 🧹 Cleanup (Save Money!) {#7-cleanup-save-money}

> IMPORTANT:
> **TPU Clusters are Expensive!**
> Always delete your cluster when you are done. Even idle clusters incur
> significant hourly costs.

By default, `launch_xpk.py` uses `--delete_cluster_on_completion=True`, which
automatically deletes the cluster after the workload finishes.

**However, you MUST manually delete the cluster if:**

*   You explicitly set `--delete_cluster_on_completion=False` (e.g., for
debugging).
*   The script crashed or was interrupted before cleanup could occur.

### Option 1: XPK (Recommended)
This is the safest method as it cleans up associated resources.

```bash
xpk cluster delete --cluster <CLUSTER_NAME> --zone <ZONE>
```

### Option 2: Gcloud CLI
If you don't have XPK handy or it is failing:

```bash
gcloud container clusters delete <CLUSTER_NAME> --zone <ZONE> --project <PROJECT_ID>
```

### Option 3: Google Cloud Console (UI)
1.  Go to the **[GKE Clusters Dashboard](https://console.cloud.google.com/kubernetes/list)**.
2.  Select your cluster (checkbox).
3.  Click **Delete** (🗑️ icon) in the top action bar.
4.  Type the cluster name to confirm deletion.