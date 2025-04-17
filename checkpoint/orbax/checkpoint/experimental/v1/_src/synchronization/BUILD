package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
)

py_library(
    name = "synchronization",
    srcs = ["synchronization.py"],
    deps = [
        ":multihost",
        ":signaling_client",
        "//checkpoint/orbax/checkpoint/_src/futures:synchronization",
    ],
)

py_library(
    name = "signaling_client",
    srcs = ["signaling_client.py"],
    deps = ["//orbax/checkpoint/_src/futures:signaling_client"],
)

py_library(
    name = "multihost",
    srcs = ["multihost.py"],
    deps = [":signaling_client"],
)
