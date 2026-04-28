load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library", pytype_strict_test = "pytype_strict_contrib_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "renaming",
    srcs = ["renaming.py"],
    visibility = [
        "//third_party/py/qwix/integration_tests:__pkg__",
        "//third_party/py/tpu_kernel_testbench:__pkg__",
    ],
    deps = [":types"],
)

py_library(
    name = "fusing",
    srcs = ["fusing.py"],
    visibility = ["//third_party/py/tpu_kernel_testbench:__pkg__"],
    deps = [":types"],
)

py_library(
    name = "stacking",
    srcs = ["stacking.py"],
    visibility = ["//third_party/py/tpu_kernel_testbench:__pkg__"],
    deps = [":types"],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = ["//orbax/checkpoint"],
)

py_library(
    name = "nesting",
    srcs = ["nesting.py"],
    visibility = [
        "//third_party/py/qwix/integration_tests:__pkg__",
        "//third_party/py/tpu_kernel_testbench:__pkg__",
    ],
    deps = [
        ":types",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_library(
    name = "repeating",
    srcs = ["repeating.py"],
    visibility = ["//third_party/py/tpu_kernel_testbench:__pkg__"],
    deps = [":types"],
)

py_test(
    name = "fusing_test",
    srcs = ["fusing_test.py"],
    deps = [":fusing"],
)

py_test(
    name = "renaming_test",
    srcs = ["renaming_test.py"],
    deps = [":renaming"],
)

py_test(
    name = "stacking_test",
    srcs = ["stacking_test.py"],
    tags = ["requires-pufferfish:4"],
    deps = [":stacking"],
)

py_test(
    name = "nesting_test",
    srcs = ["nesting_test.py"],
    deps = [":nesting"],
)

py_test(
    name = "repeating_test",
    srcs = ["repeating_test.py"],
    deps = [":repeating"],
)
