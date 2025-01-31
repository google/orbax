package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_checkpointer",
    srcs = ["abstract_checkpointer.py"],
    deps = ["//checkpoint/orbax/checkpoint:version"],
)

py_library(
    name = "checkpointer",
    srcs = ["checkpointer.py"],
    deps = [
        ":abstract_checkpointer",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint:utils",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/futures:synchronization",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_defaults",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
    ],
)

py_library(
    name = "pytree_checkpointer",
    srcs = ["pytree_checkpointer.py"],
    deps = [
        ":checkpointer",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
    ],
)

py_library(
    name = "standard_checkpointer",
    srcs = ["standard_checkpointer.py"],
    deps = [
        ":async_checkpointer",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
    ],
)

py_library(
    name = "async_checkpointer",
    srcs = ["async_checkpointer.py"],
    deps = [
        ":checkpointer",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint:utils",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/futures:synchronization",
        "//checkpoint/orbax/checkpoint/_src/handlers:async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:async_utils",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
    ],
)

py_library(
    name = "checkpointer_test_utils",
    srcs = ["checkpointer_test_utils.py"],
    deps = [
        ":async_checkpointer",
        ":checkpointer",
        "//checkpoint/orbax/checkpoint:args",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint:utils",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/handlers:async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/logging:step_statistics",
        "//checkpoint/orbax/checkpoint/_src/metadata:array_metadata",
        "//checkpoint/orbax/checkpoint/_src/metadata:array_metadata_store",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/testing:test_tree_utils",
    ],
)
