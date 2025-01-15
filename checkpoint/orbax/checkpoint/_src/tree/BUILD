package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [":types"],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//checkpoint/orbax/checkpoint/_src/testing:test_tree_utils",
    ],
)
