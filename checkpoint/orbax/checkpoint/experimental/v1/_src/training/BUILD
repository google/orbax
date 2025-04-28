package(default_visibility = ["//visibility:public"])

py_library(
    name = "save_decision_policies",
    srcs = ["save_decision_policies.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/checkpoint_managers:save_decision_policy",
        "//orbax/checkpoint/experimental/v1/_src/training/metadata:types",
    ],
)

py_library(
    name = "checkpointer",
    srcs = ["checkpointer.py"],
    deps = [
        ":errors",
        ":save_decision_policies",
        "//checkpoint/orbax/checkpoint:checkpoint_manager",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/handlers:pytree_handler",
        "//orbax/checkpoint/experimental/v1/_src/loading",
        "//orbax/checkpoint/experimental/v1/_src/metadata:loading",
        "//orbax/checkpoint/experimental/v1/_src/metadata:types",
        "//orbax/checkpoint/experimental/v1/_src/path:format_utils",
        "//orbax/checkpoint/experimental/v1/_src/path:step",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/saving",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:types",
        "//orbax/checkpoint/experimental/v1/_src/training/metadata:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
        "//orbax/checkpoint/google:pathways_type_handlers",
    ],
)

py_library(
    name = "checkpointer_test_base",
    srcs = ["checkpointer_test_base.py"],
    deps = [
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//orbax/checkpoint/experimental/v1",
        "//orbax/checkpoint/experimental/v1/_src/path:step",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:multihost",
        "//orbax/checkpoint/experimental/v1/_src/testing:array_utils",
        "//orbax/checkpoint/experimental/v1/_src/testing:handler_utils",
        "//orbax/checkpoint/experimental/v1/_src/testing:tree_utils",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
)
