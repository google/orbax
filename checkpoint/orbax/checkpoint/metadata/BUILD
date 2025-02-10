package(default_visibility = ["//visibility:public"])

py_library(
    name = "metadata",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tree",
        ":value",
        "//checkpoint/orbax/checkpoint/_src/metadata:checkpoint",
        "//checkpoint/orbax/checkpoint/_src/metadata:sharding",
        "//checkpoint/orbax/checkpoint/_src/metadata:step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
    ],
)

py_library(
    name = "value",
    srcs = ["value.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
    ],
)

py_library(
    name = "tree",
    srcs = ["tree.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/metadata:tree",
    ],
)
