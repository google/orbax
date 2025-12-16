# Base image argument (defaulting to slim python image)
ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

WORKDIR /app

# 1. Install System Dependencies
# common utils + git (needed for checkout)
# --no-install-recommends limits bloat
# python3-pip is standard in python images, no need to install
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# 2. Checkout Orbax (Optimized Shallow Fetch)
ARG PR_NUMBER
ARG BRANCH=main
ARG REPO_URL=https://github.com/google/orbax.git

# Logic:
# 1. Init empty repo
# 2. Add remote
# 3. Shallow fetch ONLY the target (PR or Branch)
# 4. Checkout
# 5. DELETE .git history to save space
RUN mkdir orbax_repo && cd orbax_repo && \
    git init && \
    git remote add origin $REPO_URL && \
    if [ -n "$PR_NUMBER" ]; then \
      echo "Fetching PR #${PR_NUMBER} (Shallow)..." && \
      git fetch --depth 1 origin pull/$PR_NUMBER/head:pr_branch && \
      git checkout pr_branch; \
    else \
      echo "Fetching branch: ${BRANCH} (Shallow)..." && \
      git fetch --depth 1 origin $BRANCH && \
      git checkout FETCH_HEAD; \
    fi && \
    rm -rf .git

WORKDIR /app/orbax_repo

# 3. Setup Python Environment & Dependencies
# Uninstall pre-installed orbax if present in base image to avoid conflicts
RUN pip uninstall -y orbax-checkpoint orbax || true

ARG JAX_VERSION=newest
ARG DEVICE=tpu

# Install GCSFS and Portpicker
RUN pip install --no-cache-dir gcsfs portpicker clu tensorflow

# Install requirements from repo root if it exists
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install JAX (Flexible Versions)
RUN if [ "$JAX_VERSION" = "newest" ]; then \
      if [ "$DEVICE" = "gpu" ]; then \
        pip install --no-cache-dir -U "jax[k8s,cuda12]" jaxlib; \
      elif [ "$DEVICE" = "tpu" ]; then \
        pip install --no-cache-dir -U "jax[k8s,tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
      else \
         pip install --no-cache-dir -U "jax[k8s]" jaxlib; \
      fi \
    elif [ "$JAX_VERSION" = "nightly" ]; then \
      if [ "$DEVICE" = "gpu" ]; then \
        pip install --no-cache-dir -U --pre "jax[k8s,cuda12]" jaxlib --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/; \
      elif [ "$DEVICE" = "tpu" ]; then \
        pip install --no-cache-dir -U --pre "jax[k8s,tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/; \
      fi \
    else \
      # Specific version
      if [ "$DEVICE" = "gpu" ]; then \
         pip install --no-cache-dir "jax[k8s,cuda12]==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"; \
      elif [ "$DEVICE" = "tpu" ]; then \
         pip install --no-cache-dir "jax[k8s,tpu]==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
      else \
         pip install --no-cache-dir "jax[k8s]==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"; \
      fi \
    fi

# 4. Install Orbax from Source
WORKDIR /app/orbax_repo/checkpoint
RUN pip install --no-cache-dir .

# 5. Environment Setup
# Set PYTHONPATH so 'import orbax' works from the correctly mapped directory
ENV PYTHONPATH=/app/orbax_repo/checkpoint

# Verify installation
RUN python3 -c "import orbax.checkpoint; print('Orbax installed:', orbax.checkpoint.__file__)"

# 6. Entrypoint
# We point to the benchmark script relative to the repo root structure
WORKDIR /app/orbax_repo/checkpoint
ENTRYPOINT ["python3", "orbax/checkpoint/_src/testing/benchmarks/run_benchmarks.py"]
