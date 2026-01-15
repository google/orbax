#!/bin/bash
# Script to build and push Orbax benchmark Docker image.

set -e

# Default values
PROJECT_ID=$(gcloud config get-value project)
IMAGE_NAME="orbax-benchmarks"
USER_TAG_FLAG=""
PR_NUMBER=""
BRANCH="main"
JAX_VERSION="newest"
DEVICE="tpu"
BASE_IMAGE=""
DOCKERFILE_PATH=""
NO_CACHE_FLAG=""

function print_usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"+
  echo "  --project PROJECT_ID    GCP Project ID"
  echo "  --pr PR_NUMBER          GitHub PR number"
  echo "  --image IMAGE_NAME      Image name (default: orbax-benchmarks)"
  echo "  --branch BRANCH         GitHub branch (default: main)"
  echo "  --jax-version VERSION   JAX version: newest, nightly, or X.Y.Z (default: newest)"
  echo "  --device DEVICE         Device type: tpu, gpu, cpu (default: tpu)"
  echo "  --base-image IMAGE      Base Docker image (optional)"
  echo "  --dockerfile FILE       Dockerfile path (optional)"
  echo "  --tag TAG               Image tag"
  echo "  --no-cache              Disable Docker build cache"
  echo "  --help                  Show this help"
}

# LINT.IfChange(build_image_flags)
# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --pr) PR_NUMBER="$2"; shift 2 ;;
    --image) IMAGE_NAME="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --jax-version) JAX_VERSION="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --dockerfile) DOCKERFILE_PATH="$2"; shift 2 ;;
    --tag) USER_TAG_FLAG="$2"; shift 2 ;;
    --no-cache) NO_CACHE_FLAG="--no-cache"; shift 1 ;;
    --help) print_usage; exit 0 ;;
    *) echo "Unknown argument: $1"; print_usage; exit 1 ;;
  esac
done
# LINT.ThenChange(README.md:build_image_flags_table)

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: Project ID not set."
  exit 1
fi

# Set default base image if not provided
if [[ -z "$BASE_IMAGE" ]]; then
  BASE_IMAGE="python:3.11-slim"
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if [[ -z "$DOCKERFILE_PATH" ]]; then
  DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile"
fi

if [[ ! -f "$DOCKERFILE_PATH" ]]; then
  # Fallback: check if we are running in the source dir
  if [[ -f "./Dockerfile" ]]; then
    DOCKERFILE_PATH="./Dockerfile"
  else
    echo "Error: Dockerfile not found at ${DOCKERFILE_PATH} or ./Dockerfile"
    exit 1
  fi
fi

# --- Tag generation ---
declare -a build_tags=()
if [[ -n "$USER_TAG_FLAG" ]]; then
  build_tags+=("$USER_TAG_FLAG")
fi
if [[ -n "$PR_NUMBER" ]]; then
  build_tags+=("pr-$PR_NUMBER")
elif [[ -n "$BRANCH" ]]; then
  build_tags+=("$BRANCH")
fi

if [[ ${#build_tags[@]} -eq 0 ]]; then
  build_tags+=("${BRANCH}-${DEVICE}-${JAX_VERSION}")
fi

# de-duplicate tags
IFS=$'\n' tags=($(sort -u <<<"${build_tags[*]}"))
unset IFS
# --- End tag generation ---

IMAGE_REPO="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"

echo "=========================================================="
echo "Building Orbax Benchmark Image"
echo "Project:   ${PROJECT_ID}"
echo "Image:     ${IMAGE_REPO}"
echo "Tags:      ${tags[*]}"
if [[ -n "$PR_NUMBER" ]]; then
  echo "Source:    PR #$PR_NUMBER"
else
  echo "Source:    Branch '$BRANCH'"
fi
echo "Base:      ${BASE_IMAGE}"
echo "JAX:       ${JAX_VERSION}"
echo "Device:    ${DEVICE}"
echo "=========================================================="

declare -a build_tag_args=()
for t in "${tags[@]}"; do
  build_tag_args+=(-t "${IMAGE_REPO}:${t}")
done

# Build with local Docker
echo "Building with previously installed Docker..."
declare -a build_args=()
if [[ -n "${NO_CACHE_FLAG}" ]]; then
  build_args+=("${NO_CACHE_FLAG}")
fi
build_args+=(
  "--build-arg" "BASE_IMAGE=${BASE_IMAGE}"
  "--build-arg" "BRANCH=${BRANCH}"
  "--build-arg" "JAX_VERSION=${JAX_VERSION}"
  "--build-arg" "DEVICE=${DEVICE}"
  "--build-arg" "PR_NUMBER=${PR_NUMBER}"
)
build_args+=("${build_tag_args[@]}")
build_args+=(
  "-f" "${DOCKERFILE_PATH}"
  "."
)
docker build "${build_args[@]}"

echo "Pushing image to registry..."
for t in "${tags[@]}"; do
  docker push "${IMAGE_REPO}:${t}"
done

echo "=========================================================="
echo "Build Success!"
echo "Image pushed to: ${IMAGE_REPO} with tags: ${tags[*]}"
echo "=========================================================="
echo "You can now run benchmarks using:"
echo "python3 launch_xpk.py --docker_image=${IMAGE_REPO}:${tags[0]} ..."
