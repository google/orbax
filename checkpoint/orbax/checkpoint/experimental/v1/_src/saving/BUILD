package(default_visibility = ["//visibility:public"])

py_library(
    name = "saving",
    srcs = ["saving.py"],
    deps = [
        "//orbax/checkpoint",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/handlers:pytree_handler",
        "//orbax/checkpoint/experimental/v1/_src/path:format_utils",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
        "//orbax/checkpoint/google:pathways_type_handlers",
    ],
)
