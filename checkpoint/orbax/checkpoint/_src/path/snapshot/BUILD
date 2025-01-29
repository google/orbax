package(default_visibility = ["//visibility:public"])

py_library(
    name = "snapshot",
    srcs = ["snapshot.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/path:utils"],
)

py_test(
    name = "snapshot_test",
    srcs = ["snapshot_test.py"],
    deps = [":snapshot"],
)
