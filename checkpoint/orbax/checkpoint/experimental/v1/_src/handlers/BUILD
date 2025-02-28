load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization:types",
    ],
)
