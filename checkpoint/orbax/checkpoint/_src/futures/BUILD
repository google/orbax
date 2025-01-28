package(default_visibility = ["//visibility:public"])

py_library(
    name = "synchronization",
    srcs = ["synchronization.py"],
)

py_test(
    name = "synchronization_test",
    srcs = ["synchronization_test.py"],
    deps = [":synchronization"],
)

py_library(
    name = "future",
    srcs = ["future.py"],
    deps = [
        ":synchronization",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)
