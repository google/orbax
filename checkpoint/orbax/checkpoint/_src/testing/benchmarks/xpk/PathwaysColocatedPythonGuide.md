# Orbax Benchmark on Pathways Colocated Python Guide

## Introduction

This guide provides step-by-step instructions for running Orbax benchmarks
on **Pathways with Colocated Python**. Before you start, make sure you are
familiarize with [README.md](README.md) on how to run `xpk`. This setup allows
running benchmarks where the Python code executes directly on the TPU worker
nodes (colocated with the Pathways server), which is useful for reducing data
transfer between Pathways Client and Workers.

## Step-by-step Instructions

### Create a Pathways cluster

Use `launch_xpk.py --create_cluster` option or command below

```shell
xpk cluster create-pathways \
  --cluster orbax-benchmark-v5p-64-pw \
  --tpu-type=v5p-64 \
  --num-slices=1 \
  --spot \
  --zone=us-east5-a
```

### Install latest XPK

```shell
pip install --upgrade xpk
```

### Build both benchmark & sidecar images.

- Use `--jax-version 0.10.1` for Benchmark image build. This has to match
  with `Dockerfile.sidecar`'s source image.
- Use `--base-image python:3.12-slim` for the Benchmark image build, the python
  version also has with `Dockerfile.sidecar`'s source image.
- Use `--build-sidecar true` to build sidecar image as well.

This is a full example of `build_image.sh` command.
```shell
export DATE_STR=(`date +"%Y%m%d-%H%M%Z"`)
export IMG_TAG=orbax-benchmark-local-tpu-$DATE_STR

bash build_image.sh  \
  --project orbax-checkpoint \
  --local-repo ./ \
  --tag $IMG_TAG \
  --device tpu \
  --jax-version 0.10.1 \
  --base-image python:3.12-slim \
  --no-cache \
  --build-benchmark true \
  --build-sidecar true
```

### Run Benchmark

Following is the full example to run a Orbax Benchmark on Pathways Colocated
Python. Make sure to set the sidecar docker_image in `--pathways_sidecar_image`
to the images you have just built above.

```shell
MODEL=llama-70b-v5p-64-pw
OUTPUT_DIR=gs://orbax-benchmarks/${USER}/${DATE_STR}/$MODEL # don't end with slash
CONFIG_DIR=$OUPTUT_DIR

python3 launch_xpk.py \
  --verbose \
  --enable_pathways \
  --cluster_name dnlng-v5p-64-pw \
  --create_cluster=False \
  --delete_cluster_on_completion=False \
  --spot \
  --tpu_type v5p-64 \
  --num_slices 1 \
  --zone us-east5-a \
  --config_file=orbax/checkpoint/_src/testing/benchmarks/configs/llama-70b-v5p-64-pw-example.yaml \
  --docker_image=gcr.io/orbax-checkpoint/orbax-benchmarks:$IMG_TAG \
  --pathways_sidecar_image=gcr.io/orbax-checkpoint/orbax-benchmarks/sidecar:$IMG_TAG \
  --output_directory=$OUTPUT_DIR \
  --config_directory=$CONFIG_DIR

```