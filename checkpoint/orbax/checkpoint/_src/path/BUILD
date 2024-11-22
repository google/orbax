load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library", pytype_strict_test = "pytype_strict_contrib_test")
load("//devtools/python/blaze:strict.bzl", "py_library_providing_imports_info")

package(
    default_applicable_licenses = ["//third_party/py/orbax/checkpoint:package_license"],
    default_visibility = [
        "//third_party/py/orbax/checkpoint:__subpackages__",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "path",
    srcs = ["__init__.py"],
    lib_rule = pytype_strict_library,
)
