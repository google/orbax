# Orbax Model CLI

A command-line tool for inspecting Orbax models.

## Examples

To inspect the model:

  ```bash
  % obm-cli show /path/to/model

  Loading Orbax model...
  Manifest: 1832 KiB, 7 objects, 2 supplementals
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Object                     ┃ Type                                ┃
  ┃                            ┃                                     ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ serving_default            │ MLIR StableHLO v1.0                 │
  │                            │ inlined: 1246605 bytes              │
  │                            │                                     │
  │ serving_default_post       │ TensorFlow ConcreteFunction v0.0.1  │
  │                            │ file: serving_default_post.pb       │
  │                            │                                     │
  │ serving_default_pre        │ TensorFlow ConcreteFunction v0.0.1  │
  │                            │ file: serving_default_pre.pb        │
  │                            │                                     │
  │ weights                    │ Orbax Checkpoint                    │
  │                            │ file: checkpoint/                   │
  │                            │                                     │
  └────────────────────────────┴─────────────────────────────────────┘
  ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Supplemental           ┃ Type                                        ┃
  ┃                        ┃                                             ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ orchestration          │ Orbax Export Orchestration Pipelines v0.0.1 │
  │                        │ file: orchestration.pb                      │
  │                        │                                             │
  │ tensorflow_saved_model │ TensorFlow SavedModel v1.0                  │
  │                        │ file: tf_saved_model/                       │
  │                        │                                             │
  └────────────────────────┴─────────────────────────────────────────────┘  
  ```

To show more details about an object or a supplemental in the model:

  ```bash
  % obm-cli show /path/to/model --details <object_or_supplemental_name>

  # E.g.
  % obm-cli show /path/to/model --details serving_default

  Loading Orbax model...
  Manifest: 1832 KiB, 7 objects, 2 supplementals
  Showing details for serving_default...
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Function                   ┃ serving_default                                         ┃
  ┃                            ┃                                                         ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ Type                       │ StableHLO                                               │
  │                            │                                                         │
  │ Visibility                 │ public                                                  │
  │                            │                                                         │
  │ Size                       │ 1246605 bytes (`--savehlo <file>` to save locally)      │
  │                            │                                                         │
  │ Input signature            │ (                                                       │
  │                            │   (                                                     │
  │                            │     {                                                   │
  │                            │       EmbeddingMapper_0/query_dense_embedder: {         │
  │                            │         b: [768, bf16],                                 │
  │                            │         w: [1024, 768, bf16]                            │
  │                            │       },                                                │
  │                            │       EmbeddingMapper_0/query_dense_hidden_0: {         │
  │                            │         b: [1024, bf16],                                │
  │                            │         w: [110, 1024, bf16]                            │
  │                            │   ...462 more lines... (`--verbose` to expand)          │
  │                            │                                                         │
  │ Output signature           │ {                                                       │
  │                            │   scores: [_, 50, bf16],                                │
  │                            │   tokens: [_, 50, 8, si32]                              │
  │                            │ }                                                       │
  │                            │                                                         │
  │ Lowering platforms         │ tpu                                                     │
  │                            │                                                         │
  │ Calling convention version │ 10                                                      │
  │                            │                                                         │
  │ Data names                 │ 160 total (`--verbose` to expand)                       │
  │                            │                                                         │
  │ jax_specific_info          │ JAX Supplemental Function v0.0.1                        │
  │                            │ file: serving_default_jax_specific_info_supplemental.pb │
  │                            │ `--jax_specific_info` to see                            │
  │                            │                                                         │
  │ xla_compile_options        │ XLA Compile Options                                     │
  │                            │ inlined: 3958 bytes                                     │
  │                            │ `--xla_compile_options` to see                          │
  │                            │                                                         │
  └────────────────────────────┴─────────────────────────────────────────────────────────┘

  ```
