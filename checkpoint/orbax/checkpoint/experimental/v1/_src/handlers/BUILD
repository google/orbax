load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library", pytype_strict_test = "pytype_strict_contrib_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:types",
    ],
)

py_library(
    name = "pytree_handler_test_base",
    srcs = ["pytree_handler_test_base.py"],
    deps = [
        ":pytree_handler",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint:utils",
        "//checkpoint/orbax/checkpoint/_src/arrays:abstract_arrays",
        "//checkpoint/orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:array_metadata",
        "//checkpoint/orbax/checkpoint/_src/metadata:array_metadata_store",
        "//checkpoint/orbax/checkpoint/_src/metadata:empty_values",
        "//checkpoint/orbax/checkpoint/_src/metadata:sharding",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//checkpoint/orbax/checkpoint/_src/serialization:replica_slices",
        "//checkpoint/orbax/checkpoint/_src/serialization:tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
        "//orbax/checkpoint/experimental/v1/_src/context:options",
        "//orbax/checkpoint/experimental/v1/_src/testing:array_utils",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
        "//orbax/checkpoint/google:pathways_type_handlers",
    ],
)

py_library(
    name = "pytree_handler",
    srcs = ["pytree_handler.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint:checkpoint_utils",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/futures:synchronization",
        "//checkpoint/orbax/checkpoint/_src/handlers:base_pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:array_metadata_store",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//orbax/checkpoint/experimental/v1/_src/context:options",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)
