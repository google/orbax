package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "types",
    srcs = ["types.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/arrays:abstract_arrays"],
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

py_library(
    name = "structure_utils",
    srcs = ["structure_utils.py"],
    deps = [
        ":parts_of",
        ":utils",
    ],
)

py_test(
    name = "structure_utils_test",
    srcs = ["structure_utils_test.py"],
    deps = [
        ":parts_of",
        ":structure_utils",
        "//checkpoint/orbax/checkpoint:test_utils",
    ],
)

py_library(
    name = "parts_of",
    srcs = ["parts_of.py"],
    deps = [":utils"],
)

py_test(
    name = "parts_of_test",
    srcs = ["parts_of_test.py"],
    deps = [":parts_of"],
)
