# Orbax - Checkpointing for JAX Models

[![PyPI version](https://img.shields.io/pypi/v/orbax-checkpoint?label=orbax-checkpoint)](https://pypi.org/project/orbax-checkpoint/)
[![PyPI version](https://img.shields.io/pypi/v/orbax-export?label=orbax-export)](https://pypi.org/project/orbax-export/)
[![Documentation Status](https://readthedocs.org/projects/orbax/badge/?version=latest)](https://orbax.readthedocs.io/en/latest/?badge=latest)
[![Build Checkpoint](https://img.shields.io/github/check-runs/google/orbax/main?nameFilter=build-checkpoint-summary&label=test-checkpoint)](https://github.com/google/orbax/actions/workflows/build.yml)
[![Build Export](https://img.shields.io/github/check-runs/google/orbax/main?nameFilter=build-export-summary&label=test-export)](https://github.com/google/orbax/actions/workflows/build.yml)
[![Checkpoint Benchmarks](https://img.shields.io/github/check-runs/google/orbax/main?nameFilter=multiprocess-checkpoint-benchmarks-summary&label=checkpoint-benchmarks)](https://github.com/google/orbax/actions/workflows/multiprocess_tests.yml)
[![Multiprocess Unit Tests](https://img.shields.io/github/check-runs/google/orbax/main?nameFilter=multiprocess-unit-tests-summary&label=checkpoint-multiprocess-unit-tests)](https://github.com/google/orbax/actions/workflows/multiprocess_tests.yml)

[**Installation**](#installation) | [**Quickstart**](#quickstart) |
[**Documentation**](https://orbax.readthedocs.io/en/latest/) |
[**Support**](#support)

Orbax provides common checkpointing and persistence utilities for JAX users.

## [Documentation](https://orbax.readthedocs.io/en/latest/)

Refer to our full documentation [here](https://orbax.readthedocs.io/en/latest/).

## Installation

Orbax is available on PyPI as separate domain-specific packages:

### Checkpointing

Install from [PyPI](https://pypi.org/project/orbax-checkpoint/):

```sh
pip install orbax-checkpoint
```

Or install the latest version directly from GitHub at HEAD:

```sh
pip install 'git+https://github.com/google/orbax/#subdirectory=checkpoint'
```

### Exporting

Install from [PyPI](https://pypi.org/project/orbax-export/):

```sh
pip install orbax-export
```

Or install the latest version directly from GitHub at HEAD:

```sh
pip install 'git+https://github.com/google/orbax/#subdirectory=export'
```

## Quickstart

```python
import jax
from orbax.checkpoint import v1 as ocp

# Define your pytree state (e.g. weights, optimizer state)
state = {'a': jax.numpy.ones(2), 'b': 42}

# Save the state
ocp.save('/tmp/my_checkpoint', state)

# Restore the state
restored_state = ocp.load('/tmp/my_checkpoint')
```

Orbax includes a checkpointing library oriented towards JAX users, supporting a
variety of different features required by different frameworks, including
asynchronous checkpointing, standard/custom types, and flexible storage formats.
We aim to provide a highly customizable and composable API which maximizes
flexibility for diverse use cases.


## Support


Please report any issues or request support using our
[issue tracker](https://github.com/google/orbax/issues).

Please also reach out to orbax-dev@google.com directly for help or with any
questions about Orbax.

## Citing Orbax

Our paper is available on [arXiv](https://arxiv.org/abs/2605.23066).

If you use Orbax in your research, please cite:

```
@misc{gaffney2026orbaxdistributedcheckpointingjax,
      title={Orbax: Distributed Checkpointing with JAX},
      author={Colin Gaffney and Shutong Li and Daniel Ng and Anastasia Petrushkina and Niket Kumar and Adam Cogdell and Mridul Sahu and Yaning Liang and Nikhil Bansal and Justin Pan and Angel Mau and Abhishek Agrawal and Marco Berlot and Ruoxin Sang and Kiranbir Sodhia and Rakesh Iyer},
      year={2026},
      eprint={2605.23066},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2605.23066},
}
```

## Existing Users

Orbax Checkpointing is used extensively across JAX machine learning frameworks
and model implementations.

### Google Projects

-   [Flax](https://github.com/google/flax) (Google's flexible and expressive
    neural network library for JAX)
-   [Gemma](https://github.com/google-deepmind/gemma) (Open foundation models by
    Google DeepMind)
-   [Kauldron](https://github.com/google-research/kauldron) (Google Research
    training and evaluation framework)
-   [PaxML](https://github.com/google/paxml) (Google's high-performance
    framework for training large-scale JAX models)
-   [T5X](https://github.com/google-research/t5x) (Google's JAX framework for
    high-performance sequence models)
-   [MaxText](https://github.com/google/maxtext) (Google's high-performance,
    scalable JAX LLM implementation)
-   [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Stable
    diffusion JAX training library optimized for Cloud TPUs)
-   [Tunix](https://github.com/google/tunix) (Google's JAX-native library for
    LLM post-training)
-   Numerous Google-internal ML frameworks

### Non-Google Projects

-   [AXLearn](https://github.com/apple/axlearn) (Apple's high-performance deep
    learning library built on top of JAX)
-   [openpi](https://github.com/physical-intelligence/openpi) (Robotics
    foundation models by Physical Intelligence)
