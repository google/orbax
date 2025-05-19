package(default_visibility = ["//visibility:public"])

py_library(
    name = "save_load_test_base",
    srcs = ["save_load_test_base.py"],
    deps = [
        ":array_utils",
        ":handler_utils",
        ":tree_utils",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
        "//third_party/py/aiofiles",
        "//orbax/checkpoint/experimental/v1",
        "//orbax/checkpoint/experimental/v1/_src/handlers:registration",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:multihost",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)

py_library(
    name = "array_utils",
    srcs = ["array_utils.py"],
    deps = [
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/arrays:abstract_arrays",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)

py_library(
    name = "handler_utils",
    srcs = ["handler_utils.py"],
    deps = [
        "//third_party/py/aiofiles",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/handlers:types",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:multihost",
    ],
)

py_library(
    name = "path_utils",
    srcs = ["path_utils.py"],
    deps = ["//orbax/checkpoint/experimental/v1/_src/path:types"],
)

py_library(
    name = "tree_utils",
    srcs = ["tree_utils.py"],
    deps = [
        "//orbax/checkpoint/experimental/v1/_src/path:format_utils",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:multihost",
    ],
)

py_library(
    name = "v0v1_compatibility_save_load_test_base",
    srcs = ["v0v1_compatibility_save_load_test_base.py"],
    deps = [
        ":array_utils",
        "//checkpoint/orbax/checkpoint:args",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:checkpointer",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:standard_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//orbax/checkpoint/experimental/v1",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:multihost",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)
