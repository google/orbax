package(default_visibility = ["//visibility:public"])

py_test(
    name = "checkpointer_test_single_worker",
    srcs = ["checkpointer_test.py"],
    args = [
        "--jax_platforms=pathways",
        "--jax_backend_target=subprocess",
        "--pathways_ifrt=true",
        "--jax_allow_unused_tpus=true",
    ],
    main = "checkpointer_test.py",
    deps = [
        "//learning/pathways/data_parallel:remote_python_support",  # build_cleaner: keep
        "//learning/pathways/data_parallel:tpu_support",  # buildcleaner: keep
        "//learning/pathways/jax:pathways_with_local_server",  # build_cleaner: keep
        "//pyglib/contrib/g3_multiprocessing",
        "//testing/pybase:parameterized",
        "//orbax/checkpoint/experimental/v1/_src/training:checkpointer_test_base",
    ],
)

py_test(
    name = "checkpointer_test_multi_worker",
    srcs = ["checkpointer_test.py"],
    args = [
        "--jax_platforms=pathways",
        "--jax_backend_target=subslice",
        "--pathways_ifrt=true",
        "--jax_allow_unused_tpus=true",
        "--pathways_expected_instances=df=1x1,df=1x1,df=1x1,df=1x1",
    ],
    main = "checkpointer_test.py",
    deps = [
        "//learning/pathways/data_parallel:remote_python_support",  # build_cleaner: keep
        "//learning/pathways/data_parallel:tpu_support",  # buildcleaner: keep
        "//learning/pathways/jax:pathways_with_local_server",  # build_cleaner: keep
        "//pyglib/contrib/g3_multiprocessing",
        "//testing/pybase:parameterized",
        "//orbax/checkpoint/experimental/v1/_src/training:checkpointer_test_base",
    ],
)
