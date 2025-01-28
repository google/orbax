package(default_visibility = ["//visibility:public"])

licenses(["notice"])

proto_library(
    name = "foo_proto",
    srcs = ["foo.proto"],
)

py_proto_library(
    name = "foo_py_pb2",
    deps = [":foo_proto"],
)
