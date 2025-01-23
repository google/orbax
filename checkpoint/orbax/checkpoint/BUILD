package(
    default_applicable_licenses = [":package_license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "checkpoint",
    srcs = ["__init__.py"],
    lib_rule = pytype_strict_library,
    visibility = ["//visibility:public"],
    deps = [
        ":abstract_checkpoint_manager",
        ":aggregate_handlers",
        ":args",
        ":arrays",
        ":checkpoint_manager",
        ":checkpoint_managers",
        ":checkpoint_utils",
        ":checkpointers",
        ":future",
        ":handlers",
        ":logging",
        ":msgpack_utils",
        ":options",
        ":path",
        ":test_utils",
        ":transform_utils",
        ":tree",
        ":type_handlers",
        ":utils",
        ":version",
        "//checkpoint/orbax/checkpoint/_src/handlers:async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//orbax/checkpoint/_src/handlers:array_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:base_pytree_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:handler_registration",
        "//orbax/checkpoint/_src/handlers:json_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:random_key_checkpoint_handler",
        "//orbax/checkpoint/metadata",
        "//orbax/checkpoint/serialization",
    ],
)

py_library(
    name = "handlers",
    srcs = ["handlers.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/handlers:async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:array_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:handler_registration",
        "//orbax/checkpoint/_src/handlers:handler_type_registry",
        "//orbax/checkpoint/_src/handlers:json_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:random_key_checkpoint_handler",
    ],
)

py_library(
    name = "checkpoint_args",
    srcs = ["checkpoint_args.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:handler_type_registry",
    ],
)

py_test(
    name = "checkpoint_args_test",
    srcs = ["checkpoint_args_test.py"],
    python_version = "PY3",
    deps = [
        ":checkpoint_args",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:handler_type_registry",
    ],
)

py_library(
    name = "args",
    srcs = ["args.py"],
    deps = [
        ":checkpoint_args",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:array_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:json_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:random_key_checkpoint_handler",
    ],
)

py_library(
    name = "abstract_checkpoint_manager",
    srcs = ["abstract_checkpoint_manager.py"],
    deps = [":args"],
)

py_library(
    name = "checkpoint_manager",
    srcs = ["checkpoint_manager.py"],
    srcs_version = "PY3",
    tags = ["ignore_for_dep=//orbax/checkpoint/google:storage_configuration_alerter"],
    deps = [
        ":abstract_checkpoint_manager",
        ":args",
        ":checkpoint_args",
        ":logging",
        ":options",
        ":utils",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:checkpointer",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
        "//checkpoint/orbax/checkpoint/_src/path:deleter",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//checkpoint/orbax/checkpoint/_src/path:utils",
        "//third_party/py/jax/experimental/array_serialization:serialization",
        "//orbax/checkpoint/_src/checkpoint_managers:save_decision_policy",
        "//orbax/checkpoint/_src/checkpointers:abstract_checkpointer",
        "//orbax/checkpoint/_src/checkpointers:async_checkpointer",
        "//orbax/checkpoint/_src/handlers:handler_registration",
        "//orbax/checkpoint/_src/handlers:json_checkpoint_handler",
        "//orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//orbax/checkpoint/_src/metadata:root_metadata_serialization",
        "//orbax/checkpoint/google:storage_configuration_alerter",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_args",
        "//checkpoint/orbax/checkpoint/_src/handlers:async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/multihost:multislice",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//checkpoint/orbax/checkpoint/_src/serialization:replica_slices",
        "//checkpoint/orbax/checkpoint/_src/serialization:tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:async_utils",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_library(
    name = "transform_utils",
    srcs = ["transform_utils.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_library(
    name = "future",
    srcs = ["future.py"],
    deps = ["//orbax/checkpoint/_src/futures:future"],
)

py_library(
    name = "aggregate_handlers",
    srcs = ["aggregate_handlers.py"],
    deps = [
        ":future",
        ":msgpack_utils",
        ":utils",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
    ],
)

py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    deps = [
        ":utils",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//orbax/checkpoint/_src/path/snapshot",
    ],
)

py_library(
    name = "msgpack_utils",
    srcs = ["msgpack_utils.py"],
    deps = ["//third_party/py/msgpack"],
)

py_test(
    name = "msgpack_utils_test",
    srcs = ["msgpack_utils_test.py"],
    deps = [":msgpack_utils"],
)

py_test(
    name = "checkpoint_utils_test",
    srcs = ["checkpoint_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":args",
        ":checkpoint_manager",
        ":checkpoint_utils",
        ":test_utils",
        ":utils",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//orbax/checkpoint/_src/checkpointers:pytree_checkpointer",
    ],
)

py_test(
    name = "transform_utils_test",
    srcs = ["transform_utils_test.py"],
    deps = [
        ":test_utils",
        ":transform_utils",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_test(
    name = "single_host_test",
    srcs = ["single_host_test.py"],
    deps = [
        ":test_utils",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//third_party/py/ml_dtypes",
        "//orbax/checkpoint/_src/handlers:standard_checkpoint_handler_test_utils",
    ],
)

py_library(
    name = "conftest",
    srcs = ["conftest.py"],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/multihost"],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)

py_library(
    name = "logging",
    srcs = ["logging.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/logging:step_statistics",
        "//orbax/checkpoint/_src/logging:abstract_logger",
        "//orbax/checkpoint/_src/logging:cloud_logger",  # buildcleaner:keep
        "//orbax/checkpoint/_src/logging:composite_logger",
        "//orbax/checkpoint/_src/logging:standard_logger",
    ],
)

py_library(
    name = "tree",
    srcs = ["tree.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/tree:types",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_library(
    name = "path",
    srcs = ["path.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/path:async_utils",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_defaults",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
        "//checkpoint/orbax/checkpoint/_src/path:deleter",
        "//checkpoint/orbax/checkpoint/_src/path:format_utils",
        "//checkpoint/orbax/checkpoint/_src/path:step",
    ],
)

py_library(
    name = "checkpointers",
    srcs = ["checkpointers.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/checkpointers:checkpointer",
        "//orbax/checkpoint/_src/checkpointers:abstract_checkpointer",
        "//orbax/checkpoint/_src/checkpointers:async_checkpointer",
        "//orbax/checkpoint/_src/checkpointers:pytree_checkpointer",
        "//orbax/checkpoint/_src/checkpointers:standard_checkpointer",
    ],
)

py_library(
    name = "type_handlers",
    srcs = ["type_handlers.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/serialization:type_handlers"],
)

py_library(
    name = "arrays",
    srcs = ["arrays.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/arrays:abstract_arrays",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
    ],
)

py_library(
    name = "checkpoint_managers",
    srcs = ["checkpoint_managers.py"],
    deps = [
        ":abstract_checkpoint_manager",
        ":checkpoint_manager",
        "//orbax/checkpoint/_src/checkpoint_managers:save_decision_policy",
    ],
)
