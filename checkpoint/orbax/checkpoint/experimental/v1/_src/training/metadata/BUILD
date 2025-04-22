load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint_info",
        "//orbax/checkpoint/experimental/v1/_src/metadata:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)
