# Orbax Checkpoint TPU VM Benchmarks


**The "Bare Metal" Developer Experience for Orbax on Cloud TPUs.**

Welcome to the Orbax TPU VM tooling. Unlike XPK, which abstracts infrastructure
behind Kubernetes/Docker for production stability, this directory provides a
**"Bare Metal"** workflow designed for **speed, visibility, and control**.

If you are a developer writing new checkpointing logic, debugging distributed
arrays, or profiling low-level performance, you are in the right place.

---

## 📚 Table of Contents
1.  [The Concept: You are the Controller](#1-the-concept-you-are-the-controller)
2.  [Prerequisites](#2-prerequisites)
3.  [Part 1: The Provisioning Story (`manage_tpu.sh`)](#3-part-1-the-provisioning-story-manage_tpush)
4.  [Part 2: The Development Loop (`launch_on_vm.sh`)](#4-part-2-the-development-loop-launch_on_vmsh)
5.  [The Control Panel: Flags & Use Cases](#5-the-control-panel-flags--use-cases)
6.  [Developer Internals](#6-developer-internals)
7.  [Troubleshooting](#7-troubleshooting)
8.  [Decision Guide](#8-environment-decision-guide)

---

## 1. The Concept: You are the Controller {#1-the-concept-you-are-the-controller}

In this architecture, there is no "Job Server" or "Cluster Controller". Your
local machine (CloudTop or Laptop) is the orchestrator. You interact directly
with the TPU workers via SSH.

**The Flow:**

1.  **You** provision raw VMs.
2.  **You** push your code and config.
3.  **You** trigger the run.
4.  **You** see the logs stream back instantly.

```sequence-diagram
participant Local_Developer
participant TPU_Workers

Note over Local_Developer,TPU_Workers: Phase 1: Initialization
Local_Developer->TPU_Workers: 1. Provision (manage_tpu.sh)
Local_Developer->TPU_Workers: 2. Setup Dependencies (setup_tpu.sh)

Note over Local_Developer,TPU_Workers: Phase 2: The Inner Loop
Local_Developer->TPU_Workers: 3. Sync Code (git fetch)
Local_Developer->TPU_Workers: 4. Push Config (SSH)
Local_Developer->TPU_Workers: 5. Run Benchmark
TPU_Workers->Local_Developer: 6. Stream Logs
```

---

## 2. Prerequisites {#2-prerequisites}

Before starting, ensure you are authenticated. The scripts rely on `gcloud` to
find and talk to TPUs.

```bash
# 1. Login to your User Account (for gcloud CLI)
gcloud auth login

# 2. Login Application Default Credentials (for Python libraries used by scripts)
gcloud auth application-default login
```

### 🔗 Useful Resources

*   **[TPU Accelerator Zones](https://docs.cloud.google.com/compute/docs/regions-zones/accelerator-zones)**

---

## 3. Part 1: The Provisioning Story (`manage_tpu.sh`) {#3-part-1-the-provisioning-story-manage_tpush}

 **"I need hardware."**

The `manage_tpu.sh` script is your inventory manager. It hides the complexity of
dealing with two different GCP APIs: the **TPU VM API** (for v4/v5p single
slices) and the **Queued Resources API** (for v5e/v6e multislice).

### Scenario A: "I just need to test my code on a TPU."
You want a small, single-host TPU. It's instant and cheap.

```bash
# Create a standard v4-8
./manage_tpu.sh create --tpu-name orbax-dev-1 --type v4-8 --zone us-central2-b
```

### Scenario B: "I need to test distributed scaling."
You need a Multislice environment (Multiple hosts connected via ICI). This
requires a "Queued Resource". Use `--node-count > 1`.

```bash
# Create a v6e-16 (2 nodes * 8 chips)
# This will poll the queue until your resource is ACTIVE.
./manage_tpu.sh create --tpu-name orbax-scale-1 --type v6e-16 --zone europe-west4-a --node-count 2
```

### Scenario C: "Is it ready yet?"
You took a coffee break. Now check the status. The script automatically figures
out if you are asking about a VM or a Queue.

```bash
./manage_tpu.sh status --tpu-name orbax-scale-1
```

---

## 4. Part 2: The Development Loop (`launch_on_vm.sh`) {#4-part-2-the-development-loop-launch_on_vmsh}

**"I have hardware. Let's run code."**

The `launch_on_vm.sh` script is your **Inner Loop**. It connects to every worker
in your TPU pod (whether 1 or 100) and executes your will.

### Step 1: The First Run (Bootstrap)
Your VMs are empty. You need Python, JAX, and Orbax installed.

*   **Action**: Use `--setup-script`.
*   **Result**: Installs `pip` packages and clones the repository (default `main`).
*   **Options**: Use `--branch=my-feature` or `--pr=123` to setup a specific version.

```bash
./launch_on_vm.sh \
  --tpu-name orbax-dev-1 \
  --config ../configs/pytree_checkpoint_benchmark.yaml \
  --setup-script setup_tpu.sh \
  --branch main # Optional
```

### Step 2: The Code Edit (The "Update" Loop)
You found a bug, or you want to review a PR.

*   **Action**: Run with `--update`.
*   **Result**: The script runs `git fetch && git checkout` on all workers. **Time: < 5 seconds.**
*   *Note: This is drastically faster than rebuilding a Docker image in XPK.*

```bash
./launch_on_vm.sh \
  --tpu-name orbax-dev-1 \
  --config ../configs/pytree_checkpoint_benchmark.yaml \
  --update \
  --pr 123 # Optional: Switch to PR #123
```

### Step 3: The Debug Session
Something is weird. The benchmark hangs. You want to see what's running on the
machine.

*   **Action**: Use `--command`.
*   **Result**: Runs your arbitrary shell command on all workers.

```bash
# Check if python processes are stuck
./launch_on_vm.sh --tpu-name orbax-dev-1 --command "ps aux | grep python"

# Kill them all
./launch_on_vm.sh --tpu-name orbax-dev-1 --command "sudo pkill -9 python3"
```

---

## 5. The Control Panel: Flags & Use Cases {#5-the-control-panel-flags--use-cases}

### 🟢 Essentials
| Flag | Description | When to use? |
| :--- | :--- | :--- |
| `--tpu-name` | Name of your TPU resource. | **Always**. |
| `--config` | Path to benchmark YAML. | **Always** (unless using `--command`). |
| `--zone` | GCP Zone (default: `europe-west4-a`). | If your TPU is not in the default zone. |

### 🔄 The Loop (Development)
| Flag | Description | When to use? |
| :--- | :--- | :--- |
| `--setup-script` | Path to `setup_tpu.sh`. | **First Run**. Bootstraps the environment. |
| `--update` | Triggers `git pull` on workers. | **Code Changes**. Syncs your latest commit. |
| `--branch` | Git branch to fetch (default: `main`). | **Feature Work**. Testing a specific Branch. |
| `--pr` | GitHub PR ID (e.g., `123`). | **Reviewing**. Fetch code from a specific PR. |
| `--repo-url` | Custom Fork URL. | **Forks**. Testing from your own fork. |
| `--jax-version` | JAX version (default: `newest`). | **Regression Testing**. Test `nightly` vs `newest`. |
| `--ramfs-dir` | Local path to save checkpoints. | **ECM Testing**. Testing restore from local. |
| `--test_restart_workflow` | If True, run workload creation and execution twice to test restart. | **ECM Testing**. To test restart behavior. |

### 🛠️ Diagnostics
| Flag | Description | When to use? |
| :--- | :--- | :--- |
| `--command` | Custom shell command. | **Debugging**. Inspecting process state or cleanup. |
| `--discover-only` | Print worker hostnames and exit. | **Connectivity Check**. Verify SSH access. |


---

## 6. Developer Internals {#6-developer-internals}

For those contributing to the scripts themselves.

### 1. Config Upload Strategy
We do not rely on GCS or NFS to share configs, as that introduces latency and
consistency issues. For simplicity, configuration is uploaded directly to each
worker via `gcloud compute tpus tpu-vm scp`. This ensures that each worker
receives the configuration with minimal delay.

*   **Method**: `SCP`.
*   **Benefit**: Reliable file transfer managed by `gcloud`.

### 2. Smart Provisioning (`manage_tpu.sh`)

*   **Problem**: Users shouldn't need to know if they are using the `tpu-vm` API
    (Single Slice) or `queued-resources` API (Multislice).
*   **Solution**: The script acts as a facade. For operations like `status` or
    `delete`, it attempts to find the resource in the QR API first. If that
    fails (404), it falls back to the VM API.
*   **Benefit**: a unified interface (`./manage_tpu.sh delete --tpu-name foo`) for all TPU types.

### 3. Robust Setup (`setup_tpu.sh`)

*   **The Race**: Freshly booted VMs often run `unattended-upgrades` in the
    background, holding the `apt` lock.
*   **The Fix**: `setup_tpu.sh` includes a busy-wait loop that checks for
    `/var/lib/dpkg/lock` using `fuser`. It waits (up to 5 minutes) for the lock
    to release before attempting installations, preventing spurious CI failures.

### 4. Lightweight Code Sync (`--update`)

*   **Concept**: Re-running the full setup takes minutes. Pushing code should
    take seconds.
*   **Mechanism**: We assume the repo exists. We run:

    ```bash
    git fetch origin branch --depth=1 && git checkout FETCH_HEAD
    ```

*   **Why Detached HEAD?**: It's safer than `git pull` which can merge/conflict. We treat the worker's code as an ephemeral mirror of your local state.


---

## 7. Troubleshooting {#7-troubleshooting}

| Symptom | Diagnosis | Fix |
| :--- | :--- | :--- |
| **`Permission denied (publickey)`** | SSH keys not propagated. | Run `gcloud auth login`. Check `--reason` if internal. |
| **`agent refused operation`** | SSH Agent overloaded. | The script handles this with staggered starts, but try killing local `ssh-agent`. |
| **`Could not get lock /var/lib/dpkg`** | VM is auto-updating. | `setup_tpu.sh` now has a retry loop. Wait 60s and try again. |
| **`jax.distributed` hangs** | Network Partition. | Ensure `dnsutils` is installed (run `--setup-script`). |

---

## 8. Environment Decision Guide: The "Where Should I Run?" Story {#8-environment-decision-guide}

Choosing the right environment is critical for your velocity.

### 1. The Definitions

*   **Single Slice**: A contiguous set of chips (e.g., v4-8, v5p-128)
    interconnected entirely by high-bandwidth **ICI** (Inter-Chip Interconnect).
    It behaves like a single supercomputer.
*   **Multislice**: Multiple slices connected over the **DCN** (Data Center
    Network). This allows scaling beyond a single Pod (e.g., thousands of
    chips). Required for massive LLM training.
*   **XPK (Accelerated Processing Kit)**: A GKE-based orchestrator that
    containerizes your workload. It is the "Production Factory" where you ship
    immutable Docker images.
*   **TPU VM Scripts (This Directory)**: The "Bare Metal" approach. You run processes directly on the host OS via SSH.

### 2. Deep Dive Comparison

| Feature | **TPU VM Scripts** (Bare Metal) | **XPK** (GKE / Containerized) |
| :--- | :--- | :--- |
| **Primary User** | **Developer / Researcher** | **Release Engineer / Production CI** |
| **Inner Loop Speed** | 🚀 **< 10s** (Git pull & run) | 🐢 **5-15 mins** (Docker build + push + pod sched) |
| **Debuggability** | 👁️ **Full** (SSH, htop, pdb, edit installed files on fly) | 📦 **Black Box** (Logs only, harder to attach debuggers) |
| **Reproducibility** | ⚠️ **Low** (Mutable state, rigid dependencies harder) | ✅ **High** (Immutable containers, exact SHA pinning) |
| **Scale** | Best for **1-4 Slices**. Managing SSH for 1000 nodes is clear but noisy. | Best for **100+ Slices**. GKE handles health checks/restarts better. |
| **Preemption** | **Manual**. You must restart the script. | **Auto-Healing**. Kueue/JobSet handles restarts automatically. |

### 3. Recommendations

#### ✅ Use TPU VM Scripts when:

1.  **You are debugging**. You need to insert `print()` statements, use `pdb`, or run `tcpdump` to trace network packets.
2.  **You are iterating**. You are changing code every 5 minutes. Waiting for Docker builds will kill your flow.
3.  **You are profiling**. You need raw access to `libtpu` counters or host OS metrics (`/sys/class/...`) without container overhead/permissions issues.

#### ✅ Use XPK when:

1.  **You are running overnight regressions**. You want the job to auto-restart if a cosmic ray hits a chip.
2.  **You are training a finalized model**. The code is stable. You need an audit trail of exactly what ran (Docker SHA).
3.  **You need massive scale**. Orchestrating 1024 workers via SSH is possible (we do it internally), but GKE's control plane is built for 10k+ nodes.

### 4. Single Slice vs. Multislice (Networking Nuance)

| | Single Slice | Multislice |
| :--- | :--- | :--- |
| **Interconnect** | **ICI Only**. Ultra-low latency connectivity. | **ICI + DCN**. Chips in a slice use ICI. Slices talk over Data Center Network. |
| **JAX Config** | `jax.distributed.initialize()` works out of the box. | Requires `jax.distributed.initialize()` + careful DCN mesh configuration. |
| **Provisioning** | `tpu-vm` API (Synchronous, fast). | `queued-resources` API (Asynchronous, supports future scheduling). |

> **Pro Tip**: Always debug on **Single Slice** first. If it crashes on v4-8, it will definitely crash on v4-16000 (Multislice). Move to Multislice only when your code is numerically correct and optimized on a single slice.
