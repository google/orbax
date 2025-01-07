package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "multihost",
    srcs = ["multihost.py"],
)

py_test(
    name = "multihost_test",
    srcs = ["multihost_test.py"],
    python_version = "PY3",
    deps = [":multihost"],
)

py_library(
    name = "multislice",
    srcs = ["multislice.py"],
    deps = [":multihost"],
)

py_test(
    name = "multislice_test",
    srcs = ["multislice_test.py"],
    python_version = "PY3",
    deps = [":multislice"],
)
