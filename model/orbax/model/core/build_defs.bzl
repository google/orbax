"""Build macro for proto fork."""

load("//devtools/build_cleaner/skylark:build_defs.bzl", "register_extension_info")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

def ml_proto_library(name, srcs = [], deps = []):
    proto_library(name = name, srcs = srcs, deps = deps)
    name_prefix = name[:name.index("_proto")]
    cc_proto_library(name = name_prefix + "_cc_proto", deps = [":" + name])
    py_proto_library(name = name_prefix + "_py_pb2", deps = [":" + name])

register_extension_info(
    extension = ml_proto_library,
    label_regex_for_dep = "{extension_name}",
)
