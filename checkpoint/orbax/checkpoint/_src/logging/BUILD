package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_logger",
    srcs = ["abstract_logger.py"],
)

py_library(
    name = "step_statistics",
    srcs = ["step_statistics.py"],
)

py_library(
    name = "cloud_logger",
    srcs = ["cloud_logger.py"],
    deps = [":abstract_logger"],
)

py_library(
    name = "standard_logger",
    srcs = ["standard_logger.py"],
    deps = [":abstract_logger"],
)

py_test(
    name = "standard_logger_test",
    srcs = ["standard_logger_test.py"],
    deps = [":standard_logger"],
)

py_library(
    name = "composite_logger",
    srcs = ["composite_logger.py"],
    deps = [":abstract_logger"],
)
