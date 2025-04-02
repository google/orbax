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
        ":signaling_client",
        ":synchronization",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_library(
    name = "signaling_client",
    srcs = ["signaling_client.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/multihost"],
)

py_test(
    name = "signaling_client_test",
    srcs = ["signaling_client_test.py"],
    deps = [
        ":signaling_client",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)
