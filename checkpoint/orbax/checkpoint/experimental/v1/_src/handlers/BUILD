package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = ["//orbax/checkpoint/experimental/v1/_src/path:types"],
)

py_test(
    name = "types_test",
    srcs = ["types_test.py"],
    deps = [
        ":types",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
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
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/context:options",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/serialization:registration",
        "//orbax/checkpoint/experimental/v1/_src/testing:array_utils",
        "//orbax/checkpoint/experimental/v1/_src/testing:path_utils",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)

py_library(
    name = "pytree_handler",
    srcs = ["pytree_handler.py"],
    deps = [
        ":registration",
        ":types",
        "//checkpoint/orbax/checkpoint:checkpoint_utils",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/futures:synchronization",
        "//checkpoint/orbax/checkpoint/_src/handlers:base_pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:array_metadata_store",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/context:options",
        "//orbax/checkpoint/experimental/v1/_src/metadata:types",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)

py_library(
    name = "registration",
    srcs = ["registration.py"],
    deps = [":types"],
)

py_library(
    name = "composite_handler",
    srcs = ["composite_handler.py"],
    deps = [
        ":global_registration",
        ":registration",
        "//checkpoint/orbax/checkpoint/_src:composite",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
    ],
)

py_library(
    name = "compatibility",
    srcs = ["compatibility.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/handlers:async_checkpoint_handler",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization",
    ],
)

py_library(
    name = "proto_handler",
    srcs = ["proto_handler.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
    ],
)

py_library(
    name = "global_registration",
    srcs = ["global_registration.py"],
    deps = [
        ":json_handler",
        ":proto_handler",
        ":pytree_handler",
        ":registration",
        "//orbax/checkpoint/experimental/v1/_src/path:format_utils",
    ],
)

py_test(
    name = "registration_test",
    srcs = ["registration_test.py"],
    deps = [
        ":registration",
        ":types",
        "//orbax/checkpoint/experimental/v1/_src/testing:handler_utils",
    ],
)

py_test(
    name = "composite_handler_test",
    srcs = ["composite_handler_test.py"],
    deps = [
        ":composite_handler",
        ":global_registration",
        ":pytree_handler",
        ":registration",
        ":types",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//orbax/checkpoint/experimental/v1/_src/testing:handler_utils",
        "//orbax/checkpoint/experimental/v1/_src/testing:path_utils",
    ],
)

py_library(
    name = "json_handler",
    srcs = ["json_handler.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)

py_test(
    name = "json_handler_test",
    srcs = ["json_handler_test.py"],
    deps = [
        ":json_handler",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/testing:path_utils",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)
