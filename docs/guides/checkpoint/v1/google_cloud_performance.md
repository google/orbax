# Checkpointing on Google Cloud

[![View on GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/google/orbax/blob/main/docs/guides/checkpoint/v1/google_cloud_performance.md)

This guide covers how to choose and configure Google Cloud storage services and
Orbax to maximize checkpoint save and load performance. It builds on the general
concepts in
[Maximizing Performance](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/maximizing_performance.html).

---

## Overview

Google Cloud Storage (GCS) **bucket capabilities** (Standard, Hierarchical
Namespace, or Rapid/Zonal) and **access paths** (GCS Direct `gs://` vs. GCSFuse
mounts) are independent choices that can be combined.

At large cluster scales, or with interruptible spot capacity, **Mean Time
Between Failures (MTBF)**—the average elapsed training time between hardware
failures, preemptions, or node interruptions—can drop substantially. Shorter
MTBF requires more frequent or
[continuous checkpointing](https://developers.googleblog.com/boost-training-goodput-how-continuous-checkpointing-optimizes-reliability-in-orbax-and-maxtext/)
(see
[Checkpointing in a Training Loop](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/training.html))
and faster storage tiers to maintain high training goodput.

| Option | Throughput & Consistency | Setup Effort | Best For |
| :--- | :--- | :--- | :--- |
| **[GCS Direct (`gs://`)](#gcs-direct-gs)** | Moderate | None | Small-to-medium jobs or standalone GCE VMs |
| **[GCSFuse](#gcsfuse)** | High | Moderate | **Default choice** for most GKE training workloads |
| **[GCS Hierarchical Namespace (HNS)](#gcs-hierarchical-namespace-hns)** | Faster metadata & higher QPS limits | Minimal | Multi-host training on either GCSFuse or `gs://` |
| **[GCS Rapid Storage (Zonal)](#gcs-rapid-storage-zonal-buckets)** | Very High | Low–Moderate | Single-zone clusters needing fast saves and loads |
| **[Managed Lustre](#google-cloud-managed-lustre)** | Highest | High | Multi-pod clusters, short MTBF, or shared multi-job storage |

---

## Recommended Orbax Settings

All examples below configure
[`ocp.Context`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.html#orbax.checkpoint.v1.Context)
(see
[Configuring Specialized Features](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/context.html)).
Only non-default or conditional settings that require action are listed below;
keep Orbax's default
[OCDBT + Zarr3 storage format](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/checkpoint_format.html)
enabled (`use_ocdbt=True`, `use_zarr3=True`).

**Colocate Compute and Storage**

*   **What to do:** Provision your checkpoint storage in the **exact same single
    region** (or **exact same zone**, when using Rapid Storage) as your TPU or
    GPU cluster.

*   **What to avoid:** Do **not** use Multi-Region (`US`, `EU`, `ASIA`) or
    Dual-Region GCS buckets for active training checkpoints.

```{dropdown} Why?
Multi-region and dual-region buckets add cross-region network round-trips to
directory preparation, step listing, and checkpoint overwrites or deletions,
while incurring cross-region replication and egress charges with no benefit for
transient training steps. Checkpointing throughput can also be highly variable,
leading to potential training interruptions or necessitating infrequent
checkpointing. Keep active training checkpoints in a colocated single-region or
zonal bucket, and copy only milestone or final checkpoints cross-region if
disaster recovery is required.
```

**Asynchronous Saving, Timeouts, and Overwrites**

*   **Use asynchronous save APIs:** Asynchronous checkpointing is always
    preferred—call
    [`ocp.save_async`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.html#orbax.checkpoint.v1.save_async),
    [`ocp.save_checkpointables_async`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.html#orbax.checkpoint.v1.save_checkpointables_async),
    or
    [`ocp.training.Checkpointer`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.training.checkpointer.html#orbax.checkpoint.experimental.v1.training.Checkpointer)
    (see
    [Asynchronous Checkpointing](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/async_checkpointing.html)).

    ```{dropdown} Why?
    Asynchronous saving copies device arrays to host memory and performs
    serialization and I/O in background threads while accelerator training steps
    proceed, hiding write latency behind computation.
    ```

*   **`ctx.asynchronous.timeout_secs = 1200`** *(Default: `600`)*

    ```{dropdown} Why?
    Large multi-host saves to GCS can encounter occasional straggler tail spikes
    exceeding 10 minutes. Increasing the timeout to 20 to 30 minutes prevents
    healthy background saves from aborting prematurely.
    ```

*   **`ctx.file.skip_sync_file_validations = True`** *(Default: `False`; use
    with caution)*

    ```{dropdown} Why?
    Moves destination existence checks and overwrite directory cleanup
    (`rmtree`) off the main training thread into the background
    directory-creation thread, eliminating tens of seconds of step-blocking GCS
    metadata round-trips when overwriting existing step paths.

    *Caution:* Deferring these checks to the background prevents failing fast
    when saving duplicate checkpoints (e.g., with `overwrite=False`) or hitting
    path validation errors. Those errors will still surface, just later (when
    starting a subsequent save or explicitly awaiting completion), potentially
    leading to wasted training time.
    ```

**Save Atomicity**

*   **On GCSFuse mounts only:**
    **`ctx.atomicity.mode = ocp.options.AtomicityMode.COMMIT_FILE`** *(Default
    `AUTO` uses `ATOMIC_RENAME` on mounted paths)*

    ```{dropdown} Why?
    Because GCSFuse mounts appear as local POSIX paths, `AtomicityMode.AUTO`
    defaults to finalizing checkpoints via directory rename (`ATOMIC_RENAME`).
    On flat GCS buckets, renaming a directory copies and deletes every child
    object sequentially, stalling for minutes or failing with
    `too many open files`.

    Setting `COMMIT_FILE` finalizes checkpoints by writing a lightweight
    `commit_success.txt` marker instead, avoiding rename stalls and making
    checkpoints loadable interchangeably via `/gcsfuse/...` and `gs://...`. If
    you are migrating older GCSFuse checkpoints saved before enabling
    `COMMIT_FILE`, create an empty `commit_success.txt` file inside each valid
    legacy step directory.
    ```

**File Size, Chunking, and Compression**

*   **On GCSFuse mounts:**
    **`ctx.array.saving.ocdbt_target_data_file_size = 512 * 1024**2`** (256 MiB
    – 512 MiB; *Default on mounted paths: `2 GiB`*)

    ```{dropdown} Why?
    On mounted POSIX paths, Orbax defaults `ocdbt_target_data_file_size` to
    `2 GiB` (see
    [Checkpoint Format Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/checkpoint_format.html)).
    On GCSFuse, `256 MiB` to `512 MiB` OCDBT data files increase parallel
    upload/download streams and bound over-fetching if GCSFuse File Cache is
    enabled.
    ```

*   **Conditional (resharding on load or multi-GiB shards):**
    **`ctx.array.saving.storage_options.chunk_byte_size = 64 * 1024**2`**
    (32 MiB – 256 MiB, must not exceed `ocdbt_target_data_file_size`)

    ```{dropdown} Why?
    Splits large per-device shards into smaller storage chunks to avoid read
    amplification when
    [loading with resharding](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/maximizing_performance.html#loading-with-resharding)
    onto a different topology and to increase sub-shard I/O concurrency.
    ```

*   **Conditional (CPU-bound writes on ultra-fast storage):**
    **`ctx.array.saving.use_compression = False`** on Managed Lustre or Rapid
    Storage *(Default: `True`)*

    ```{dropdown} Why?
    Zstd compression reduces network transfer volume over Regional GCS, but
    `bfloat16` model weights often compress modestly while consuming host CPU.
    On high-bandwidth Managed Lustre or Rapid Storage, disabling compression can
    improve save throughput when host CPU saturates before storage bandwidth.
    ```

**Multi-Replica Saving and Loading**

*   **Conditional (multi-replica meshes at scale):**
    **`ctx.array.loading.use_load_and_broadcast = True`** and
    **`ctx.array.loading.load_and_broadcast_options.replica_axis_index = <axis>`**
    *(Default: `False`; see
    [Scalable Loading (Load+Broadcast Mode)](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/maximizing_performance.html#scalable-loading-load-broadcast-mode))*

    ```{dropdown} Why?
    Only a single primary model replica reads from storage and broadcasts
    weights to remaining replicas over the accelerator/data-center network,
    cutting storage read traffic and QPS proportionally to the number of
    replicas.

    Enable this **only** when both conditions hold:

    1. Your global device mesh has a data-parallel replica axis across which
       model weights are fully replicated (cannot be used on single-replica
       purely sharded meshes), and `replica_axis_index` is set to that mesh
       axis.
    2. You are running at sufficient replica scale (typically 4+
       replicas/slices) where storage read throughput, GCS QPS, or GCSFuse File
       Cache read amplification is the bottleneck. At small replica counts
       (e.g., 2 replicas), or on high-throughput Managed Lustre instances where
       storage is less of a bottleneck, broadcast overhead is more noticeable
       and larger scales may be needed before enabling it is beneficial.
    ```

*   **Conditional (GCSFuse with File Cache enabled):**
    **`ctx.array.saving.use_replica_parallel = False`** *(Default on TPU:
    `True`; see
    [Load-Balanced Saving (Replica-Parallel Mode)](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/maximizing_performance.html#load-balanced-saving-replica-parallel-mode))*

    ```{dropdown} Why?
    When `use_replica_parallel=True`, each replica writes small slices of
    replicated weights into its own OCDBT files. On restore (without
    `use_load_and_broadcast`), hosts read small slices across many hosts' OCDBT
    files; if GCSFuse File Cache is enabled, GCSFuse prefetches every touched
    file in full, causing severe read amplification and substantially slower
    restores. At high replica counts on any GCS path, you can also tune
    `min_slice_bytes_for_replica_parallel` or `max_replicas_for_replica_parallel`
    so tiny slices do not inflate GCS write QPS.
    ```

**Checkpoint Deletion and Garbage Collection**

*   **Use `ocp.delete_async(path)`** or preservation policies in
    [`ocp.training.Checkpointer`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.training.checkpointer.html#orbax.checkpoint.experimental.v1.training.Checkpointer)
    *(see
    [Checkpointing in a Training Loop](https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/training.html))*

    ```{dropdown} Why?
    **Deletion performance caveat:** Checkpoint deletion runs from a **single
    primary controller host (`primary_host=0`)**, which must scan subdirectories
    and issue network `DELETE` or `unlink` operations across thousands of OCDBT
    and metadata files written by all hosts in the cluster. Even with
    `delete_async` executing in a background thread (and multi-threaded
    unlinking on mounted filesystems), **deleting a multi-TB checkpoint on GCS
    or GCSFuse still takes tens of seconds to several minutes**.

    When using `ocp.training.Checkpointer`, **if a background deletion is still
    in progress when the next save step triggers, the next save blocks until the
    prior deletion finishes** so uncollected checkpoints do not accumulate
    unboundedly. Ensure your checkpoint save interval exceeds `save duration +
    delete duration`.
    ```

*   **On Regional `gs://` buckets (especially with HNS):**
    **`ctx.deletion.gcs_deletion_options.todelete_full_path = 'trash'`** *(path
    relative to bucket root)*

    ```{dropdown} Why?
    Instead of issuing thousands of slow per-object `DELETE` RPCs from the
    primary host, Orbax relocates expired checkpoints under
    `gs://<bucket>/trash/` (an instant atomic folder rename on HNS buckets).
    Pair this with a
    [GCS Object Lifecycle Management](https://cloud.google.com/storage/docs/lifecycle)
    rule on prefix `trash/` (`Delete` after 1 day) so GCS purges expired objects
    asynchronously.

    *Caution on Rapid Storage (Zonal buckets):* Soft-deleted files under
    `trash/` continue to count against your per-zone `RAPID` byte quota until
    asynchronous lifecycle cleanup runs hours later. Avoid relying solely on
    `todelete_full_path` if your zonal byte quota is tight.
    ```

---

## GCS Direct (`gs://`)

### When to Use It

*   You want **zero infrastructure setup**—pass a `gs://` URI directly to Orbax
    without configuring a Kubernetes CSI driver, FUSE sidecar, or mount point.

*   You are running **small-to-medium scale jobs** or standalone GCE VMs.

*   **Trade-off:** Above roughly 100+ nodes, direct `gs://` writes exhibit
    higher tail-latency variance and slower metadata finalization than a tuned
    GCSFuse mount.

### Google Cloud Setup

1.  Create a **Single-Region** GCS bucket in the **same region** as your compute
    cluster (optionally add `--enable-hierarchical-namespace`; see
    [GCS Hierarchical Namespace](#gcs-hierarchical-namespace-hns)) and grant IAM
    access to your workload service account:

    ```bash
    gcloud storage buckets create gs://MY_BUCKET \
      --location=COMPUTE_REGION \
      --uniform-bucket-level-access

    gcloud storage buckets add-iam-policy-binding gs://MY_BUCKET \
      --member="serviceAccount:WORKLOAD_SA@PROJECT_ID.iam.gserviceaccount.com" \
      --role="roles/storage.objectUser"
    ```

2.  *(Optional, if using `todelete_full_path='trash'`)* Add a
    [GCS Lifecycle rule](https://cloud.google.com/storage/docs/lifecycle) to
    purge soft-deleted checkpoints under `trash/`:

    ```bash
    cat > lifecycle.json << 'EOF'
    {
      "rule": [
        {
          "action": {"type": "Delete"},
          "condition": {"age": 1, "matchesPrefix": ["trash/"]}
        }
      ]
    }
    EOF
    gcloud storage buckets update gs://MY_BUCKET --lifecycle-file=lifecycle.json
    ```

### Orbax Setup

```python
from orbax.checkpoint import v1 as ocp

ctx = ocp.Context()
ctx.asynchronous.timeout_secs = 1200
ctx.file.skip_sync_file_validations = True
ctx.deletion.gcs_deletion_options.todelete_full_path = 'trash'

with ctx:
  response = ocp.save_async('gs://MY_BUCKET/run1/step_100', state)
```

---

## GCSFuse

### When to Use It

*   **Recommended default for medium-to-large GKE clusters** backed by Regional
    or Zonal GCS.

*   Provides substantially higher save and load throughput and tighter tail
    latency than `gs://` at scale.

### Google Cloud Setup

GCSFuse and GKE Pod networking involve several interdependent settings; follow
the official Google Cloud documentation rather than configuring mount flags from
scratch:

*   [Cloud Storage FUSE Performance Tuning Best Practices](https://cloud.google.com/storage/docs/cloud-storage-fuse/performance)

*   [Sample GKE YAML Files for Checkpointing](https://cloud.google.com/storage/docs/cloud-storage-fuse/sample-yaml-files)

*   [Cloud Storage FUSE Automated Configurations & Profiles](https://cloud.google.com/storage/docs/cloud-storage-fuse/automated-configurations)

*   [GKE Cloud Storage FUSE CSI Driver Setup](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)

When applying those guides for Orbax checkpointing, ensure the following:

*   **Use modern GKE & GCSFuse with the checkpointing profile:** Use GKE 1.36+
    (Linux 6.12+) and GCSFuse 3.9+ with the `aiml-checkpointing` profile,
    **streaming writes** (`write:enable-streaming-writes:true`), **Kernel
    Reader** (`file-system:enable-kernel-reader:true`), and
    `metadata-cache:negative-ttl-secs:0` so hosts immediately see newly written
    files. Scope the mount with `only-dir` and leave sidecar CPU limits
    unthrottled.

*   **Choose Kernel Reader vs. File Cache carefully (GPU vs. TPU):** With Kernel
    Reader enabled on GKE 1.36+, disable GCSFuse File Cache
    (`file-cache:max-size-mb:0`). On older GKE versions using File Cache +
    Parallel Downloads, back the cache with **Local NVMe SSD on GPU VMs**, and
    on **TPU VMs** (which lack bundled Local SSD) either disable File Cache or
    back it with an in-memory `emptyDir: {medium: Memory}` volume while
    budgeting host RAM.

*   **Configure `hostNetwork: true` DNS and Workload Identity:** Multi-host
    TPU/GPU Pods using `hostNetwork: true` should set
    `dnsPolicy: ClusterFirstWithHostNet` with GKE NodeLocal DNSCache enabled,
    and either set `volumeAttributes.hostNetworkPodKSA: "true"` on the
    PersistentVolume or grant bucket IAM roles to the GCE node service account.

```{warning}
Never set `file-system:kernel-list-cache-ttl-secs:-1` on a read-write checkpoint
mount. Kernel directory-list caching is intended for immutable training
datasets; on a checkpoint directory, worker nodes will cache stale directory
listings and fail to see newly saved steps or `commit_success.txt` markers
written by other hosts.
```

### Orbax Setup

```python
from orbax.checkpoint import v1 as ocp

ctx = ocp.Context()
ctx.atomicity.mode = ocp.options.AtomicityMode.COMMIT_FILE
ctx.array.saving.ocdbt_target_data_file_size = 512 * 1024**2
ctx.asynchronous.timeout_secs = 1200
ctx.file.skip_sync_file_validations = True
# If GCSFuse File Cache is enabled on TPU, also set:
# ctx.array.saving.use_replica_parallel = False

with ctx:
  response = ocp.save_async('/mnt/gcsfuse/step_100', state)
```

---

## GCS Hierarchical Namespace (HNS)

Hierarchical Namespace (HNS) is an **optional** bucket-level feature (not
required for small jobs) that works independently with **both**
[GCSFuse](#gcsfuse) and [GCS Direct (`gs://`)](#gcs-direct-gs).

### When to Use It

*   **Small jobs:** Standard flat GCS buckets are sufficient, particularly when
    using `AtomicityMode.COMMIT_FILE` and `OCDBT`.

*   **Medium-to-large multi-host jobs or frequent saves:** Enable HNS to gain
    **~8x higher initial read and write QPS limits** (preventing burst request
    throttling when hundreds of hosts checkpoint simultaneously) and **instant
    atomic folder renames** (accelerating `todelete_full_path` soft-deletes and
    directory cleanup).

### Google Cloud Setup

HNS **must** be enabled when creating a new bucket; existing flat buckets cannot
be upgraded in-place. See
[Hierarchical Namespace Overview](https://cloud.google.com/storage/docs/hns-overview)
and
[Create Buckets with Hierarchical Namespace](https://cloud.google.com/storage/docs/create-hns-bucket):

```bash
gcloud storage buckets create gs://MY_HNS_BUCKET \
  --location=COMPUTE_REGION \
  --uniform-bucket-level-access \
  --enable-hierarchical-namespace

# Grant both object and HNS folder permissions:
gcloud storage buckets add-iam-policy-binding gs://MY_HNS_BUCKET \
  --member="serviceAccount:WORKLOAD_SA@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectUser"
gcloud storage buckets add-iam-policy-binding gs://MY_HNS_BUCKET \
  --member="serviceAccount:WORKLOAD_SA@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.folderAdmin"
```

### Orbax Setup

*   Use either the [GCS Direct](#gcs-direct-gs) or [GCSFuse](#gcsfuse)
    [`ocp.Context`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.html#orbax.checkpoint.v1.Context)
    configuration above.

*   Avoid custom worker-side loops that poll or list the checkpoint root
    directory from every process; let
    [`ocp.training.Checkpointer`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.training.checkpointer.html#orbax.checkpoint.experimental.v1.training.Checkpointer)
    coordinate step discovery.

---

## GCS Rapid Storage (Zonal Buckets)

Rapid Storage is an **independent** single-zone GCS storage class and placement
option that works with **either** [GCSFuse](#gcsfuse) or
[GCS Direct (`gs://`)](#gcs-direct-gs).

### When to Use It

*   Choose Rapid Storage for **single-zone clusters** when Regional GCS save or
    restore latency bottlenecks training goodput, and you want **substantially
    faster loads and higher write throughput** while keeping
    **pay-for-stored-bytes** pricing (instead of paying 24/7 for provisioned
    Managed Lustre capacity).

*   **Trade-offs:** Data is stored within a **single zone** (unavailable during
    a zone outage) and is subject to **per-zone `RAPID` storage byte quotas**.

### Google Cloud Setup

Create a Zonal `RAPID` bucket in the **exact same zone** as your TPU/GPU nodes
(Zonal buckets require uniform bucket-level access and HNS). See the official
[Rapid Storage / Zonal Buckets documentation](https://cloud.google.com/storage/docs/rapid/rapid-bucket):

```bash
gcloud storage buckets create gs://MY_ZONAL_BUCKET \
  --location=COMPUTE_REGION \
  --placement=COMPUTE_ZONE \
  --default-storage-class=RAPID \
  --uniform-bucket-level-access \
  --enable-hierarchical-namespace
```

*   **Verify zone quota upfront:** Check your project's `RAPID` storage byte
    quota in `COMPUTE_ZONE` before launching multi-TB runs.

*   **Mount via GCSFuse or use `gs://`:** Follow [GCSFuse](#gcsfuse) (allowing
    GCSFuse to use default gRPC rather than `http1` for Zonal buckets) or pass
    `gs://MY_ZONAL_BUCKET/...` directly.

*   *(Optional)* Asynchronously copy milestone or final checkpoints to a
    Standard Regional GCS bucket if cross-zone durability is needed.

### Orbax Setup

*   Apply the [GCSFuse](#gcsfuse) or [GCS Direct](#gcs-direct-gs)
    [`ocp.Context`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.html#orbax.checkpoint.v1.Context)
    settings above, and enforce strict checkpoint retention (via
    `preservation_policies` in
    [`ocp.training.Checkpointer`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.training.checkpointer.html#orbax.checkpoint.experimental.v1.training.Checkpointer)
    or explicit `ocp.delete_async` calls) so accumulated steps do not exhaust
    your zonal byte quota.

*   Do **not** rely solely on `todelete_full_path = 'trash'` + GCS Lifecycle
    rules when zonal quota is tight, because soft-deleted files in `trash/`
    continue to consume zonal byte quota until asynchronous lifecycle cleanup
    runs.

---

## Google Cloud Managed Lustre

### When to Use It

*   Choose Managed Lustre for **very large multi-pod clusters**, high-frequency
    [continuous checkpointing](https://developers.googleblog.com/boost-training-goodput-how-continuous-checkpointing-optimizes-reliability-in-orbax-and-maxtext/)
    when cluster MTBF is short, or **shared multi-job clusters** where several
    concurrent workloads amortize one filesystem.

*   Delivers the **highest sustained write bandwidth** and **most consistent
    save times** on Google Cloud, billed continuously on **100% of provisioned
    TiB capacity**.

### Google Cloud Setup

Follow the official
[Managed Lustre CSI Driver on GKE guide](https://cloud.google.com/kubernetes-engine/docs/how-to/managed-lustre-csi)
for VPC peering, `StorageClass`, and PVC setup, keeping three operational rules
in mind:

1.  **Size provisioned capacity for target bandwidth:** Aggregate throughput
    scales linearly with `Provisioned TiB × Performance Tier` (e.g., `500` or
    `1000 MBps/TiB`). Provision enough TiB to meet your target checkpoint write
    bandwidth, not just stored checkpoint volume.

2.  **Colocate in the same VPC, region, and zone:** Provision the Lustre
    instance in the same VPC network and zone as your GKE nodes, and enable
    `--enable-lustre-csi-driver` on the cluster (add
    `--enable-legacy-lustre-port` only when connecting to a legacy port `6988`
    instance).

3.  **Serialize milestone exports to GCS:**
    `gcloud lustre instances export-data` and `import-data` allow **only 1
    active transfer job per Lustre instance at a time** (concurrent requests
    abort immediately). Export a parent directory in one batch or queue
    milestone exports sequentially.

### Orbax Setup

```python
from orbax.checkpoint import v1 as ocp

ctx = ocp.Context()
ctx.asynchronous.timeout_secs = 1200
ctx.file.skip_sync_file_validations = True
# Optional: disable Zstd compression if host CPU bottlenecks Lustre write speed:
# ctx.array.saving.use_compression = False

with ctx:
  response = ocp.save_async('/mnt/lustre/run1/step_100', state)
```

```{note}
**Using `use_load_and_broadcast` with Managed Lustre:** `use_load_and_broadcast`
has primarily been tested and calibrated for GCS, where storage throughput is
typically the bottleneck. With high-throughput Managed Lustre, inter-slice
broadcasting overhead may become more noticeable relative to storage reads, so
you may need to reach larger multi-replica scales before `use_load_and_broadcast`
becomes useful.
```
