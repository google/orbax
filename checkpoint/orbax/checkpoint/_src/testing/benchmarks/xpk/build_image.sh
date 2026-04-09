#!/bin/bash
# Script to build and push Orbax benchmark Docker image.

set -e

# Default values
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
PROJECT_ID=""
IMAGE_NAME="orbax-benchmarks"
USE_LOCAL_ORBAX="false"
USER_TAG_FLAG=""
PR_NUMBER=""
BRANCH="main"
BRANCH_SPECIFIED="false"
JAX_VERSION="newest"
DEVICE="tpu"
BASE_IMAGE=""
DOCKERFILE_PATH=""
NO_CACHE_FLAG=""
LOCAL_REPO_PATH=""

function print_usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --project PROJECT_ID    GCP Project ID"
  echo "  --pr PR_NUMBER          GitHub PR number"
  echo "  --image IMAGE_NAME      Image name (default: orbax-benchmarks)"
  echo "  --branch BRANCH         GitHub branch (default: main)"
  echo "  --jax-version VERSION   JAX version: newest, nightly, or X.Y.Z (default: newest)"
  echo "  --device DEVICE         Device type: tpu, gpu, cpu (default: tpu)"
  echo "  --base-image IMAGE      Base Docker image (optional)"
  echo "  --dockerfile FILE       Dockerfile path (optional)"
  echo "  --tag TAG               Image tag"
  echo "  --local-repo PATH       Path to local Orbax repository"
  echo "  --no-cache              Disable Docker build cache"
  echo "  --help                  Show this help"
}

function prepare_local_orbax() {
  local repo_path="$1"
  local build_context="$2"
  
  if [[ -z "$repo_path" ]]; then
    echo "Error: --local-repo path not specified."
    return 1
  fi

  local abs_path="$repo_path"
  abs_path=$(realpath "$abs_path")
  echo "Resolved local repo path: ${abs_path}"


  echo "Copying local repo contents to build context..."
  mkdir -p "$build_context/checkpoint"
  if [[ -f "${abs_path}/checkpoint/pyproject.toml" ]]; then
    echo "Found pyproject.toml in checkpoint subdirectory. Copying contents of checkpoint/."
    cp -r "${abs_path}"/checkpoint/* "$build_context/checkpoint/"
  else
    echo "Error: pyproject.toml not found in local repo."
    exit 1
  fi
  return 0
}

# LINT.IfChange(build_image_flags)
# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --pr) PR_NUMBER="$2"; shift 2 ;;
    --image) IMAGE_NAME="$2"; shift 2 ;;
    --branch) BRANCH="$2"; BRANCH_SPECIFIED="true"; shift 2 ;;
    --jax-version) JAX_VERSION="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --dockerfile) DOCKERFILE_PATH="$2"; shift 2 ;;
    --tag) USER_TAG_FLAG="$2"; shift 2 ;;
    --local-repo) LOCAL_REPO_PATH="$2"; shift 2 ;;
    --no-cache) NO_CACHE_FLAG="--no-cache"; shift 1 ;;
    --help) print_usage; exit 0 ;;
    *) echo "Unknown argument: $1"; print_usage; exit 1 ;;
  esac
done
# LINT.ThenChange(README.md:build_image_flags_table)

# Validate that only one of --pr, --branch, or --local-repo is specified
count=0
[[ -n "$PR_NUMBER" ]] && count=$((count + 1))
[[ "$BRANCH_SPECIFIED" == "true" ]] && count=$((count + 1))
[[ -n "$LOCAL_REPO_PATH" ]] && count=$((count + 1))

if [[ $count -gt 1 ]]; then
  echo "Error: Only one of --pr, --branch, or --local-repo can be specified."
  exit 1
fi

if [[ -z "$PROJECT_ID" ]]; then
  # Try to get project ID from gcloud, but don't fail if it's not set
  PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
fi

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: Project ID not set. Use --project <PROJECT_ID> or 'gcloud config set project <PROJECT_ID>'."
  exit 1
fi

# Set default base image if not provided
if [[ -z "$BASE_IMAGE" ]]; then
  BASE_IMAGE="python:3.13-slim"
fi

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

# Create a temporary directory to act as the clean build context
BUILD_CONTEXT=$(mktemp -d)
echo "Build context: ${BUILD_CONTEXT}"
# Ensure the temporary directory is cleaned up when the script exits (success or fail)
trap 'rm -rf "$BUILD_CONTEXT"' EXIT
cp "${DOCKERFILE_PATH}" "$BUILD_CONTEXT/Dockerfile"

if [[ -n "$LOCAL_REPO_PATH" ]]; then
  USE_LOCAL_ORBAX="true"
  if ! prepare_local_orbax "$LOCAL_REPO_PATH" "$BUILD_CONTEXT"; then
    exit 1
  fi
fi

cd "$BUILD_CONTEXT"

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
  "--build-arg" "USE_LOCAL_ORBAX=${USE_LOCAL_ORBAX}"
)
build_args+=("${build_tag_args[@]}")
build_args+=(
  "-f" "Dockerfile"
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
