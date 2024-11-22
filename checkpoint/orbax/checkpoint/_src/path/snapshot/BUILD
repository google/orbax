load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/orbax/checkpoint:package_license"],
    default_visibility = [
        "//learning/deepmind/jax/roc:__subpackages__",
        "//third_party/py/orbax/checkpoint:__subpackages__",
    ],
)

py_library(
    name = "snapshot",
    srcs = ["snapshot.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:timer",
        "//third_party/py/orbax/checkpoint/_src/path:utils",
    ],
)

py_test(
    name = "snapshot_test",
    srcs = ["snapshot_test.py"],
    srcs_version = "PY3",
    deps = [
        ":snapshot",
        "//absl/testing:absltest",
    ],
)
