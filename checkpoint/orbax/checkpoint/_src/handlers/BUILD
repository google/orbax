package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "checkpoint_handler",
    srcs = ["checkpoint_handler.py"],
)

py_library(
    name = "composite_checkpoint_handler",
    srcs = ["composite_checkpoint_handler.py"],
    deps = [
        ":async_checkpoint_handler",
        ":checkpoint_handler",
        ":handler_registration",
        ":proto_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src:composite",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_defaults",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
        "//orbax/checkpoint/_src/futures:future",
    ],
)

py_test(
    name = "composite_checkpoint_handler_test",
    srcs = ["composite_checkpoint_handler_test.py"],
    deps = [
        ":checkpoint_handler",
        ":composite_checkpoint_handler",
        ":handler_registration",
        ":json_checkpoint_handler",
        ":proto_checkpoint_handler",
        ":standard_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:args",
        "//checkpoint/orbax/checkpoint:logging",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:step",
    ],
)

py_library(
    name = "pytree_checkpoint_handler",
    srcs = ["pytree_checkpoint_handler.py"],
    srcs_version = "PY3",
    deps = [
        ":async_checkpoint_handler",
        ":base_pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:aggregate_handlers",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint:transform_utils",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/metadata:empty_values",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//checkpoint/orbax/checkpoint/_src/serialization:tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/tree:types",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
        "//orbax/checkpoint:utils",
        "//orbax/checkpoint/_src/futures:future",
        "//orbax/checkpoint/_src/metadata:array_metadata_store",
    ],
)

py_library(
    name = "base_pytree_checkpoint_handler",
    srcs = ["base_pytree_checkpoint_handler.py"],
    srcs_version = "PY3",
    deps = [
        ":async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/metadata:empty_values",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:format_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//checkpoint/orbax/checkpoint/_src/serialization:tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/serialization:types",
        "//checkpoint/orbax/checkpoint/_src/tree:types",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
        "//orbax/checkpoint:utils",
        "//orbax/checkpoint/_src/futures:future",
        "//orbax/checkpoint/_src/futures:synchronization",
        "//orbax/checkpoint/_src/metadata:array_metadata_store",
    ],
)

py_library(
    name = "json_checkpoint_handler",
    srcs = ["json_checkpoint_handler.py"],
    deps = [
        ":async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//orbax/checkpoint:utils",
        "//orbax/checkpoint/_src/futures:future",
    ],
)

py_library(
    name = "async_checkpoint_handler",
    srcs = ["async_checkpoint_handler.py"],
    deps = [
        ":checkpoint_handler",
        "//orbax/checkpoint/_src/futures:future",
    ],
)

py_library(
    name = "array_checkpoint_handler",
    srcs = ["array_checkpoint_handler.py"],
    deps = [
        ":async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:aggregate_handlers",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//orbax/checkpoint:utils",
        "//orbax/checkpoint/_src/futures:future",
    ],
)

py_test(
    name = "json_checkpoint_handler_test",
    srcs = ["json_checkpoint_handler_test.py"],
    python_version = "PY3",
    deps = [
        ":json_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
    ],
)

py_library(
    name = "proto_checkpoint_handler",
    srcs = ["proto_checkpoint_handler.py"],
    deps = [
        ":async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//orbax/checkpoint:utils",
        "//orbax/checkpoint/_src/futures:future",
    ],
)

py_library(
    name = "standard_checkpoint_handler",
    srcs = ["standard_checkpoint_handler.py"],
    deps = [
        ":async_checkpoint_handler",
        ":pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint:checkpoint_utils",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/metadata:pytree_metadata_options",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/tree:types",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
        "//orbax/checkpoint/_src/futures:future",
    ],
)

# BEGIN OSS-INTERNAL
# TODO(yaning): Expose this test in OSS.
py_test(
    name = "proto_checkpoint_handler_test",
    srcs = ["proto_checkpoint_handler_test.py"],
    python_version = "PY3",
    deps = [":proto_checkpoint_handler"],
)

py_library(
    name = "standard_checkpoint_handler_test_utils",
    srcs = ["standard_checkpoint_handler_test_utils.py"],
    deps = [
        ":standard_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//orbax/checkpoint:utils",
    ],
)

py_library(
    name = "random_key_checkpoint_handler",
    srcs = ["random_key_checkpoint_handler.py"],
    deps = [
        ":array_checkpoint_handler",
        ":async_checkpoint_handler",
        ":composite_checkpoint_handler",
        ":json_checkpoint_handler",
        ":pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//orbax/checkpoint/_src/futures:future",
    ],
)

py_test(
    name = "random_key_checkpoint_handler_test",
    srcs = ["random_key_checkpoint_handler_test.py"],
    python_version = "PY3",
    deps = [
        ":composite_checkpoint_handler",
        ":random_key_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:args",
    ],
)

py_library(
    name = "handler_registration",
    srcs = ["handler_registration.py"],
    deps = [
        ":checkpoint_handler",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
    ],
)

py_test(
    name = "handler_registration_test",
    srcs = ["handler_registration_test.py"],
    deps = [
        ":checkpoint_handler",
        ":handler_registration",
        ":standard_checkpoint_handler",
        "//checkpoint/orbax/checkpoint:checkpoint_args",
    ],
)

py_library(
    name = "handler_type_registry",
    srcs = ["handler_type_registry.py"],
    deps = [":checkpoint_handler"],
)

py_test(
    name = "handler_type_registry_test",
    srcs = ["handler_type_registry_test.py"],
    deps = [
        ":checkpoint_handler",
        ":handler_type_registry",
        ":standard_checkpoint_handler",
    ],
)
