package(default_visibility = ["//visibility:public"])

py_library(
    name = "loading",
    srcs = ["loading.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/checkpointers:async_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/handlers:composite_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_registration",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/handlers:compatibility",
        "//orbax/checkpoint/experimental/v1/_src/handlers:composite_handler",
        "//orbax/checkpoint/experimental/v1/_src/handlers:global_registration",
        "//orbax/checkpoint/experimental/v1/_src/metadata:types",
        "//orbax/checkpoint/experimental/v1/_src/path:format_utils",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/serialization:registration",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)
