package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "checkpoint",
    srcs = ["__init__.py"],
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
        "//checkpoint/orbax/checkpoint/_src/handlers:array_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:base_pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_registration",
        "//checkpoint/orbax/checkpoint/_src/handlers:json_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:random_key_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//checkpoint/orbax/checkpoint/metadata",
        "//checkpoint/orbax/checkpoint/serialization",
    ],
)

py_library(
    name = "handlers",
    srcs = ["handlers.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/handlers:array_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:async_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_registration",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_type_registry",
        "//checkpoint/orbax/checkpoint/_src/handlers:json_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:random_key_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
    ],
)

py_library(
    name = "checkpoint_args",
    srcs = ["checkpoint_args.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_type_registry",
    ],
)

py_test(
    name = "checkpoint_args_test",
    srcs = ["checkpoint_args_test.py"],
    deps = [
        ":checkpoint_args",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_type_registry",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
    ],
)

py_library(
    name = "args",
    srcs = ["args.py"],
    deps = [
        ":checkpoint_args",
        "//checkpoint/orbax/checkpoint/_src/handlers:array_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:json_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:random_key_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:standard_checkpoint_handler",
    ],
)

py_library(
    name = "abstract_checkpoint_manager",
    srcs = ["abstract_checkpoint_manager.py"],
    deps = [
        ":args",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
    ],
)

py_library(
    name = "checkpoint_manager",
    srcs = ["checkpoint_manager.py"],
    deps = [
        ":abstract_checkpoint_manager",
        ":args",
        ":checkpoint_args",
        ":logging",
        ":options",
        ":utils",
        "//checkpoint/orbax/checkpoint/_src/checkpoint_managers:save_decision_policy",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:abstract_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:async_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:checkpointer",
        "//checkpoint/orbax/checkpoint/_src/handlers:checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_registration",
        "//checkpoint/orbax/checkpoint/_src/handlers:json_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:proto_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint_info",
        "//checkpoint/orbax/checkpoint/_src/metadata:root_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
        "//checkpoint/orbax/checkpoint/_src/path:deleter",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//checkpoint/orbax/checkpoint/_src/path:utils",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
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
    deps = ["//checkpoint/orbax/checkpoint/_src/futures:future"],
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
        "//checkpoint/orbax/checkpoint/_src/path/snapshot",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
    ],
)

py_library(
    name = "msgpack_utils",
    srcs = ["msgpack_utils.py"],
)

py_test(
    name = "msgpack_utils_test",
    srcs = ["msgpack_utils_test.py"],
    deps = [":msgpack_utils"],
)

py_test(
    name = "checkpoint_manager_options_test",
    srcs = ["checkpoint_manager_options_test.py"],
    deps = [":checkpoint"],
)

py_test(
    name = "checkpoint_utils_test",
    srcs = ["checkpoint_utils_test.py"],
    deps = [
        ":args",
        ":checkpoint_manager",
        ":checkpoint_utils",
        ":test_utils",
        ":utils",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:pytree_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//checkpoint/orbax/checkpoint/_src/path:step",
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
        "//checkpoint/orbax/checkpoint/_src/logging:abstract_logger",
        "//checkpoint/orbax/checkpoint/_src/logging:cloud_logger",  # buildcleaner:keep
        "//checkpoint/orbax/checkpoint/_src/logging:composite_logger",
        "//checkpoint/orbax/checkpoint/_src/logging:standard_logger",
        "//checkpoint/orbax/checkpoint/_src/logging:step_statistics",
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
        "//checkpoint/orbax/checkpoint/_src/checkpointers:abstract_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:async_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:checkpointer",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:pytree_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:standard_checkpointer",
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
        "//checkpoint/orbax/checkpoint/_src/checkpoint_managers:save_decision_policy",
    ],
)
