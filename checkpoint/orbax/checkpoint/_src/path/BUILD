package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "path",
    srcs = ["__init__.py"],
)

py_library(
    name = "step",
    srcs = ["step.py"],
)

py_test(
    name = "step_test",
    srcs = ["step_test.py"],
    deps = [
        ":atomicity",
        ":step",
    ],
)

py_library(
    name = "deleter",
    srcs = ["deleter.py"],
    deps = [":step"],
)

py_test(
    name = "deleter_test",
    srcs = ["deleter_test.py"],
    deps = [
        ":deleter",
        ":step",
    ],
)

py_library(
    name = "async_utils",
    srcs = ["async_utils.py"],
    deps = [":step"],
)

py_library(
    name = "atomicity",
    srcs = ["atomicity.py"],
    deps = [
        ":async_utils",
        ":atomicity_types",
        ":path",
        ":step",
        ":utils",
    ],
)

py_test(
    name = "atomicity_test",
    srcs = ["atomicity_test.py"],
    deps = [
        ":atomicity",
        ":atomicity_types",
        ":step",
    ],
)

py_library(
    name = "atomicity_types",
    srcs = ["atomicity_types.py"],
)

py_library(
    name = "atomicity_defaults",
    srcs = ["atomicity_defaults.py"],
    deps = [
        ":atomicity",
        ":atomicity_types",
        ":step",
    ],
)

py_library(
    name = "format_utils",
    srcs = ["format_utils.py"],
)

py_test(
    name = "format_utils_test",
    srcs = ["format_utils_test.py"],
    deps = [":format_utils"],
)
