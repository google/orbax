package(default_visibility = ["//visibility:public"])

py_library(
    name = "tree_rich_types",
    srcs = ["tree_rich_types.py"],
    deps = [
        ":pytree_metadata_options",
        ":value_metadata_entry",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_test(
    name = "tree_rich_types_test",
    srcs = ["tree_rich_types_test.py"],
    deps = [
        ":tree_rich_types",
        "//checkpoint/orbax/checkpoint/_src/testing:test_tree_utils",
    ],
)

py_library(
    name = "tree",
    srcs = ["tree.py"],
    deps = [
        ":empty_values",
        ":pytree_metadata_options",
        ":tree_rich_types",
        ":value",
        ":value_metadata_entry",
        "//checkpoint/orbax/checkpoint/_src:asyncio_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:types",
        "//checkpoint/orbax/checkpoint/_src/tree:types",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_test(
    name = "tree_test",
    srcs = ["tree_test.py"],
    deps = [
        ":tree",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/serialization:types",
        "//checkpoint/orbax/checkpoint/_src/testing:test_tree_utils",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_library(
    name = "value",
    srcs = ["value.py"],
    deps = [
        ":sharding",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
    ],
)

py_library(
    name = "sharding",
    srcs = ["sharding.py"],
)

py_test(
    name = "sharding_test",
    srcs = ["sharding_test.py"],
    deps = [":sharding"],
)

py_library(
    name = "checkpoint",
    srcs = ["checkpoint.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src:composite",
        "//checkpoint/orbax/checkpoint/_src/logging:step_statistics",
    ],
)

py_test(
    name = "checkpoint_test",
    srcs = ["checkpoint_test.py"],
    deps = [
        ":checkpoint",
        ":root_metadata_serialization",
        ":step_metadata_serialization",
        "//checkpoint/orbax/checkpoint/_src/logging:step_statistics",
    ],
)

py_library(
    name = "root_metadata_serialization",
    srcs = ["root_metadata_serialization.py"],
    deps = [
        ":checkpoint",
        ":metadata_serialization_utils",
    ],
)

py_library(
    name = "step_metadata_serialization",
    srcs = ["step_metadata_serialization.py"],
    deps = [
        ":checkpoint",
        ":metadata_serialization_utils",
        "//checkpoint/orbax/checkpoint/_src/logging:step_statistics",
    ],
)

py_library(
    name = "metadata_serialization_utils",
    srcs = ["metadata_serialization_utils.py"],
)

py_library(
    name = "pytree_metadata_options",
    srcs = ["pytree_metadata_options.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/tree:utils"],
)

py_library(
    name = "value_metadata_entry",
    srcs = ["value_metadata_entry.py"],
    deps = [
        ":empty_values",
        ":pytree_metadata_options",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/serialization:types",
    ],
)

py_library(
    name = "empty_values",
    srcs = ["empty_values.py"],
    deps = [
        ":pytree_metadata_options",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_test(
    name = "empty_values_test",
    srcs = ["empty_values_test.py"],
    deps = [
        ":empty_values",
        ":pytree_metadata_options",
        "//checkpoint/orbax/checkpoint/_src/testing:test_tree_utils",
    ],
)

py_library(
    name = "array_metadata",
    srcs = ["array_metadata.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/arrays:types"],
)

py_library(
    name = "array_metadata_store",
    srcs = ["array_metadata_store.py"],
    deps = [
        ":array_metadata",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/serialization:types",
    ],
)

py_test(
    name = "array_metadata_store_test",
    srcs = ["array_metadata_store_test.py"],
    deps = [
        ":array_metadata",
        ":array_metadata_store",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
    ],
)

py_library(
    name = "checkpoint_info",
    srcs = ["checkpoint_info.py"],
)

py_test(
    name = "checkpoint_info_test",
    srcs = ["checkpoint_info_test.py"],
    deps = [":checkpoint_info"],
)
