package(default_visibility = ["//visibility:public"])

py_library(
    name = "registration",
    srcs = ["registration.py"],
    tags = ["nofixdeps"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)

py_library(
    name = "array_leaf_handler",
    srcs = ["array_leaf_handler.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint:utils",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/metadata:sharding",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/google:pathways_type_handlers",
    ],
)

py_library(
    name = "compatibility",
    srcs = ["compatibility.py"],
    deps = [
        ":array_leaf_handler",
        ":types",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/metadata:sharding",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/serialization:types",
        "//orbax/checkpoint/experimental/v1/_src/context",
        "//orbax/checkpoint/experimental/v1/_src/path:types",
        "//orbax/checkpoint/experimental/v1/_src/synchronization",
        "//orbax/checkpoint/experimental/v1/_src/tree:types",
    ],
)
