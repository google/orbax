package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
)

py_library(
    name = "async_responses",
    srcs = ["async_responses.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/futures:synchronization",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_library(
    name = "synchronization",
    srcs = ["synchronization.py"],
    deps = [
        ":signaling_client",
        "//checkpoint/orbax/checkpoint/_src/futures:synchronization",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_library(
    name = "signaling_client",
    srcs = ["signaling_client.py"],
    deps = ["//orbax/checkpoint/_src/futures:signaling_client"],
)
