package(default_visibility = ["//visibility:public"])

py_library(
    name = "save_decision_policy",
    srcs = ["save_decision_policy.py"],
    deps = [
        ":policy_checkpoint_info",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//orbax/checkpoint/_src/futures:signaling_client",
    ],
)

py_library(
    name = "preservation_policy",
    srcs = ["preservation_policy.py"],
    deps = [":policy_checkpoint_info"],
)

py_test(
    name = "preservation_policy_test",
    srcs = ["preservation_policy_test.py"],
    deps = [
        ":preservation_policy",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint_info",
    ],
)

py_library(
    name = "policy_checkpoint_info",
    srcs = ["policy_checkpoint_info.py"],
)
