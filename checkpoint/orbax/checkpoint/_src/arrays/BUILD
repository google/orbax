package(default_visibility = ["//visibility:public"])

py_library(
    name = "types",
    srcs = ["types.py"],
    visibility = ["//learning/gemini/gemax/core/checkpointing:__subpackages__"],
)

py_library(
    name = "numpy_utils",
    srcs = ["numpy_utils.py"],
    deps = [":types"],
)

py_test(
    name = "numpy_utils_test",
    srcs = ["numpy_utils_test.py"],
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
    deps = [":fragments"],
)

py_library(
    name = "subchunking",
    srcs = ["subchunking.py"],
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
    deps = [
        ":fragments",
        ":subchunking",
        ":types",
    ],
)
