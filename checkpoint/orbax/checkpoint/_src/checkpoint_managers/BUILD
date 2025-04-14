package(default_visibility = ["//visibility:public"])

py_library(
    name = "save_decision_policy",
    srcs = ["save_decision_policy.py"],
    deps = [
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint_info",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_library(
    name = "preservation_policy",
    srcs = ["preservation_policy.py"],
    deps = [
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint_info",
    ],
)

py_test(
    name = "preservation_policy_test",
    srcs = ["preservation_policy_test.py"],
    deps = [
        ":preservation_policy",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint_info",
    ],
)
