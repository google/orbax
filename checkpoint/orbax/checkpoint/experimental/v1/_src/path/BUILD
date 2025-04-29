package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
)

py_library(
    name = "step",
    srcs = ["step.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/path:step"],
)

py_library(
    name = "format_utils",
    srcs = ["format_utils.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/path:format_utils",
    ],
)

py_test(
    name = "format_utils_test",
    srcs = ["format_utils_test.py"],
    deps = [
        ":format_utils",
        "//orbax/checkpoint/experimental/v1/_src/handlers:global_registration",
        "//orbax/checkpoint/experimental/v1/_src/saving",
        "//orbax/checkpoint/experimental/v1/_src/testing:array_utils",
    ],
)
