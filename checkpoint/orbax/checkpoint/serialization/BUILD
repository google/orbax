package(default_visibility = ["//visibility:public"])

py_library(
    name = "serialization",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization",
    ],
)

py_library(
    name = "tensorstore_utils",
    srcs = ["tensorstore_utils.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/serialization:tensorstore_utils"],
)
