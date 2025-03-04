package(default_visibility = ["//visibility:public"])

py_library(
    name = "context",
    srcs = ["context.py"],
)

py_test(
    name = "context_test",
    srcs = ["context_test.py"],
    deps = [
        "//orbax/checkpoint/experimental/v1",
        "//orbax/checkpoint/experimental/v1/_src/context",
    ],
)
