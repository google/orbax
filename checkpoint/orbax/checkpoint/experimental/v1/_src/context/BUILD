package(default_visibility = ["//visibility:public"])

py_library(
    name = "context",
    srcs = ["context.py"],
    deps = [
        ":options",
        "//orbax/checkpoint/experimental/v1/_src/synchronization",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:multihost",
    ],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    deps = [
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
        "//orbax/checkpoint/experimental/v1/_src/handlers:registration",
        "//orbax/checkpoint/experimental/v1/_src/handlers:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)

py_test(
    name = "context_test",
    srcs = ["context_test.py"],
    deps = [
        ":context",
        ":options",
        "//orbax/checkpoint/experimental/v1",
    ],
)
