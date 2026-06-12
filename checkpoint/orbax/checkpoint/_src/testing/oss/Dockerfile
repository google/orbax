# Base image argument (defaulting to slim python image)
ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

WORKDIR /app

# 1. Install System Dependencies (including Google Cloud CLI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    dnsutils \
    curl \
    ca-certificates \
    gnupg \
    apt-transport-https \
    gettext-base \
    gawk \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    google-cloud-cli \
    google-cloud-cli-gke-gcloud-auth-plugin \
    kubectl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./checkpoint/orbax/checkpoint/_src/testin[g] /app/orbax_repo/checkpoint/orbax/checkpoint/_src/testing

# 3. Setup Python Environment & Dependencies
# Uninstall pre-installed orbax if present in base image to avoid conflicts
RUN pip uninstall -y orbax-checkpoint orbax || true

# Force upgrade dependencies to fix security vulnerabilities
RUN pip install -U --no-cache-dir \
    "wheel>=0.46.2" \
    "aiohttp>=3.13.4" \
    "urllib3>=2.6.3" \
    "cryptography>=46.0.7" \
    "requests>=2.33.0"

# # Create a fake docker binary that always returns success (exit 0)
# RUN echo '#!/bin/bash\nexit 0' > /usr/local/bin/docker && chmod +x /usr/local/bin/docker

# Install requirements from repo root if it exists
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

RUN pip install --no-cache-dir gcsfs portpicker clu tensorflow pyyaml

# 4. Install Orbax from Source
WORKDIR /app/orbax_repo/checkpoint
RUN pip install xpk==v1.0.0

# 5. Environment Setup
# Set PYTHONPATH so 'import orbax' works from the correctly mapped directory
ENV PYTHONPATH=/app/orbax_repo/checkpoint


CMD ["python3", "orbax/checkpoint/_src/testing/oss/cloud_run_integration_tests.py"]
