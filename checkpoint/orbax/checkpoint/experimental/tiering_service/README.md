# Orbax Checkpoint Tiering Service (CTS)

The Checkpoint Tiering Service (CTS) is an experimental feature within Orbax
designed to manage machine learning checkpoints across different storage tiers
(e.g., fast ephemeral storage and durable persistent storage). It provides APIs
to coordinate checkpoint data movement and lifecycle management.

> [!WARNING]
> This project is under **heavy development**. APIs, database schemas, and
> behavior are unstable and subject to change without notice.

> [!CAUTION]
> **Do not use this service unless you know exactly what you are doing** and
> are prepared for breaking changes. It is not intended for general use at this
> time.
