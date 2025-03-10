load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = ["//orbax/checkpoint/experimental/v1/_src/tree:types"],
)

py_library(
    name = "loading",
    srcs = ["loading.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:checkpointer",
        "//orbax/checkpoint:handlers",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/handlers:pytree_handler",
        "//orbax/checkpoint/experimental/v1/_src/path:format_utils",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
    ],
)

py_test(
    name = "loading_test",
    srcs = ["loading_test.py"],
    deps = [
        ":loading",
        ":types",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//orbax/checkpoint/experimental/v1/_src/saving",
        "//orbax/checkpoint/experimental/v1/_src/testing:array_utils",
    ],
)
