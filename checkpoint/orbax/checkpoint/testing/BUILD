package(default_visibility = ["//visibility:public"])

filegroup(
    name = "aggregate_checkpoints",
    srcs = glob([
        "aggregate_all.checkpoint/**/*",
        "aggregate_all.real/**/*",
        "aggregate_some.checkpoint/**/*",
    ]),
)

py_library(
    name = "testing",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/testing:test_tree_utils",
    ],
)
