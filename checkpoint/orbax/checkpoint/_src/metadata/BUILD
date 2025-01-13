package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "tree_rich_types",
    srcs = ["tree_rich_types.py"],
    deps = [
        ":pytree_metadata_options",
        ":value_metadata_entry",
    ],
)

py_test(
    name = "tree_rich_types_test",
    srcs = ["tree_rich_types_test.py"],
    deps = [":tree_rich_types"],
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
    ],
)

py_test(
    name = "tree_test",
    srcs = ["tree_test.py"],
    deps = [":tree"],
)

py_library(
    name = "value",
    srcs = ["value.py"],
    deps = [":sharding"],
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
)

py_test(
    name = "checkpoint_test",
    srcs = ["checkpoint_test.py"],
    deps = [
        ":checkpoint",
        ":root_metadata_serialization",
        ":step_metadata_serialization",
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
    ],
)

py_library(
    name = "metadata_serialization_utils",
    srcs = ["metadata_serialization_utils.py"],
)

py_library(
    name = "pytree_metadata_options",
    srcs = ["pytree_metadata_options.py"],
)

py_library(
    name = "value_metadata_entry",
    srcs = ["value_metadata_entry.py"],
    deps = [
        ":empty_values",
        ":pytree_metadata_options",
    ],
)

py_library(
    name = "empty_values",
    srcs = ["empty_values.py"],
    deps = [":pytree_metadata_options"],
)

py_test(
    name = "empty_values_test",
    srcs = ["empty_values_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":empty_values",
        ":pytree_metadata_options",
    ],
)
