package(default_visibility = ["//visibility:public"])

py_library(
    name = "saving",
    srcs = ["saving.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/checkpointers:async_checkpointer",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//orbax/checkpoint",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
        "//orbax/checkpoint/google:pathways_type_handlers",
    ],
)
