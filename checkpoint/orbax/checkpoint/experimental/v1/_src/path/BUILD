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
        "//checkpoint/orbax/checkpoint/_src/path:step",
    ],
)

py_library(
    name = "async_utils",
    srcs = ["async_utils.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/futures:synchronization",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity_types",
        "//orbax/checkpoint/experimental/v1/_src/context",
    ],
)

py_test(
    name = "async_utils_test",
    srcs = ["async_utils_test.py"],
    deps = [
        ":async_utils",
        "//checkpoint/orbax/checkpoint/_src/path:atomicity",
    ],
)

py_test(
    name = "format_utils_test",
    srcs = ["format_utils_test.py"],
    deps = [
        ":format_utils",
        "//checkpoint/orbax/checkpoint/_src/checkpointers:standard_checkpointer",
        "//orbax/checkpoint/experimental/v1/_src/saving",
        "//orbax/checkpoint/experimental/v1/_src/testing:array_utils",
    ],
)
