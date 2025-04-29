load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library", pytype_strict_test = "pytype_strict_contrib_test")

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
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/handlers:global_registration",
        "//orbax/checkpoint/experimental/v1/_src/loading",
        "//orbax/checkpoint/experimental/v1/_src/path:format_utils",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
    ],
)

py_test(
    name = "loading_test",
    srcs = ["loading_test.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//orbax/checkpoint/experimental/v1",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/context:options",
        "//orbax/checkpoint/experimental/v1/_src/handlers:global_registration",
        "//orbax/checkpoint/experimental/v1/_src/testing:array_utils",
        "//orbax/checkpoint/experimental/v1/_src/testing:handler_utils",
    ],
)
