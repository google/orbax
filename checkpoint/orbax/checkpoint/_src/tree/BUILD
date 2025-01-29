package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src/arrays:abstract_arrays",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/testing:test_tree_utils",
    ],
)
