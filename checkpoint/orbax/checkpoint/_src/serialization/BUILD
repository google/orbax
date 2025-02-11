package(default_visibility = ["//visibility:public"])

py_library(
    name = "tensorstore_utils",
    srcs = ["tensorstore_utils.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/arrays:subchunking",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/metadata:array_metadata",
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        ":serialization",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/metadata:empty_values",
        "//checkpoint/orbax/checkpoint/_src/metadata:pytree_metadata_options",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
    ],
)

py_library(
    name = "type_handlers",
    srcs = ["type_handlers.py"],
    deps = [
        ":replica_slices",
        ":serialization",
        ":tensorstore_utils",
        ":types",
        "//checkpoint/orbax/checkpoint/_src/arrays:subchunking",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
        "//checkpoint/orbax/checkpoint/_src/metadata:array_metadata_store",
        "//checkpoint/orbax/checkpoint/_src/metadata:empty_values",
        "//checkpoint/orbax/checkpoint/_src/metadata:sharding",
        "//checkpoint/orbax/checkpoint/_src/metadata:value",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/multihost:multislice",
        "//checkpoint/orbax/checkpoint/_src/path:async_utils",
        "//checkpoint/orbax/checkpoint/_src/path:format_utils",
    ],
)

py_test(
    name = "tensorstore_utils_test",
    srcs = ["tensorstore_utils_test.py"],
    deps = [
        ":serialization",
        ":tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/arrays:subchunking",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    deps = [
        ":replica_slices",
        ":tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/arrays:fragments",
        "//checkpoint/orbax/checkpoint/_src/arrays:numpy_utils",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_library(
    name = "replica_slices",
    srcs = ["replica_slices.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/arrays:fragments",
        "//checkpoint/orbax/checkpoint/_src/arrays:numpy_utils",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    deps = [
        ":serialization",
        ":tensorstore_utils",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/futures:future",
    ],
)

py_test(
    name = "replica_slices_test",
    srcs = ["replica_slices_test.py"],
    deps = [":replica_slices"],
)
