package(default_visibility = ["//visibility:public"])

py_test(
    name = "save_load_test",
    srcs = ["save_load_test.py"],
    args = [
        "--jax_platforms=pathways",
        "--jax_backend_target=subprocess",
        "--pathways_ifrt=true",
        "--jax_allow_unused_tpus=true",
    ],
    deps = [
        "//learning/pathways/data_parallel:remote_python_support",  # build_cleaner: keep
        "//learning/pathways/data_parallel:tpu_support",  # buildcleaner: keep
        "//learning/pathways/jax:pathways_with_local_server",  # build_cleaner: keep
        "//pyglib/contrib/g3_multiprocessing",
        "//testing/pybase:parameterized",
        "//orbax/checkpoint/experimental/v1/_src/testing:save_load_test_base",
    ],
)
