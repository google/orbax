package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
)

py_library(
    name = "numpy_utils",
    srcs = ["numpy_utils.py"],
    srcs_version = "PY3",
    deps = [":types"],
)

py_test(
    name = "numpy_utils_test",
    srcs = ["numpy_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":numpy_utils"],
)

py_library(
    name = "fragments",
    srcs = ["fragments.py"],
    deps = [
        ":numpy_utils",
        ":types",
    ],
)

py_test(
    name = "fragments_test",
    srcs = ["fragments_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":fragments"],
)

py_library(
    name = "subchunking",
    srcs = ["subchunking.py"],
    srcs_version = "PY3",
    deps = [
        ":fragments",
        ":types",
    ],
)

py_library(
    name = "abstract_arrays",
    srcs = ["abstract_arrays.py"],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src/metadata:sharding",
    ],
)

py_test(
    name = "abstract_arrays_test",
    srcs = ["abstract_arrays_test.py"],
    deps = [
        ":abstract_arrays",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
    ],
)

py_test(
    name = "subchunking_test",
    srcs = ["subchunking_test.py"],
    args = ["--vmodule=subchunking=1"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fragments",
        ":subchunking",
        ":types",
    ],
)
