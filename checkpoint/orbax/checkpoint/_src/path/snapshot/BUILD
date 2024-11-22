package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "snapshot",
    srcs = ["snapshot.py"],
    default_visibility = ["//visibility:public"],
    srcs_version = "PY3",
    deps = ["//orbax/checkpoint/orbax/checkpoint/_src/path:utils"],
)

py_test(
    name = "snapshot_test",
    srcs = ["snapshot_test.py"],
    default_visibility = ["//visibility:public"],
    srcs_version = "PY3",
    deps = [
        ":snapshot",
        "//absl/testing:absltest",
    ],
)
