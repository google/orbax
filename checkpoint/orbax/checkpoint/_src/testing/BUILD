package(default_visibility = ["//visibility:public"])

py_library(
    name = "multiprocess_test",
    testonly = 1,
    srcs = ["multiprocess_test.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/multihost"],
)

py_library(
    name = "test_tree_utils",
    srcs = ["test_tree_utils.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
        "//checkpoint/orbax/checkpoint/_src/metadata:tree_rich_types",
    ],
)
