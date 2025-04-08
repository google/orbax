package(default_visibility = ["//visibility:public"])

py_library(
    name = "saving",
    srcs = ["saving.py"],
    deps = [
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:async_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_registration",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/handlers:compatibility",
        "//orbax/checkpoint/experimental/v1/_src/handlers:global_registration",
        "//orbax/checkpoint/experimental/v1/_src/handlers:pytree_handler",
        "//orbax/checkpoint/experimental/v1/_src/handlers:registration",
        "//orbax/checkpoint/experimental/v1/_src/path:format_utils",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
        "//orbax/checkpoint/google:pathways_type_handlers",
    ],
)
