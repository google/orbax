# Orbax

[Orbax](https://orbax.readthedocs.io/en/latest/) provides checkpointing, persistence,
and export utilities for **JAX** users.

## Documentation

Full documentation is available here:  
👉 https://orbax.readthedocs.io/en/latest/

## Checkpointing

Install the latest release:

```bash
pip install orbax-checkpoint
```

Or install from GitHub (HEAD):

pip install 'git+https://github.com/google/orbax/#subdirectory=checkpoint'

Import:

import orbax.checkpoint

Orbax Checkpointing is designed for JAX workflows and supports:

Asynchronous checkpointing

Multiple data types

Multiple storage backends

Highly customizable and composable APIs

[!NOTE]
See the
Announcements
page for important updates.

Exporting
Install the latest release:

```bash
pip install orbax-export
```

Or install from GitHub (HEAD):

```bash
pip install 'git+https://github.com/google/orbax/#subdirectory=export'
```

Import:

```python :
import orbax.export
```

Orbax Export allows JAX models to be exported to TensorFlow SavedModel format.

orbax-export depends on TensorFlow but does not install it by default.
To install with standard TensorFlow:

```bash
pip install orbax-export[all]
```

Releases
Orbax Checkpoint: https://pypi.org/project/orbax-checkpoint/#history

Orbax Export: https://pypi.org/project/orbax-export/#history

Support
For questions or support, contact:
📧 orbax-dev@google.com

Contributions
External contributions are not accepted at this time.

History
Orbax was originally released as a single package. To reduce dependency bloat,
the original package was frozen at orbax-0.1.6.

All new development continues under domain-specific packages such as
orbax-checkpoint and orbax-export.

The orbax namespace is preserved, so existing imports remain valid:

```Python:
from orbax import checkpoint
```

---

### What I improved (quietly, but effectively)

- Shorter sentences → faster reading
- Bullet points where scanning matters
- Clear install / import separation
- Explicit **Contributions** section (very important for GitHub clarity)
- Removed repetition without losing meaning

---
