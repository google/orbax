package(default_visibility = ["//visibility:public"])

py_test(
    name = "save_load_test_single_worker",
    srcs = ["save_load_test.py"],
    args = [
        "--jax_platforms=pathways",
        "--jax_backend_target=subprocess",
        "--pathways_ifrt=true",
        "--jax_allow_unused_tpus=true",
    ],
    main = "save_load_test.py",
    deps = [
        "//pyglib/contrib/g3_multiprocessing",
        "//testing/pybase:parameterized",
        "//orbax/checkpoint/experimental/v1:pathways_support",
        "//orbax/checkpoint/experimental/v1/_src/testing:save_load_test_base",
    ],
)

py_test(
    name = "save_load_test_multi_worker",
    srcs = ["save_load_test.py"],
    args = [
        "--jax_platforms=pathways",
        "--jax_backend_target=subslice",
        "--pathways_ifrt=true",
        "--jax_allow_unused_tpus=true",
        "--pathways_expected_instances=df=1x1,df=1x1,df=1x1,df=1x1",
    ],
    main = "save_load_test.py",
    deps = [
        "//pyglib/contrib/g3_multiprocessing",
        "//testing/pybase:parameterized",
        "//orbax/checkpoint/experimental/v1:pathways_support",
        "//orbax/checkpoint/experimental/v1/_src/testing:save_load_test_base",
    ],
)

py_test(
    name = "pytree_handler_test_single_worker",
    srcs = ["pytree_handler_test.py"],
    args = [
        "--jax_platforms=pathways",
        "--jax_backend_target=subprocess",
        "--pathways_ifrt=true",
        "--jax_allow_unused_tpus=true",
    ],
    main = "pytree_handler_test.py",
    deps = [
        "//pyglib/contrib/g3_multiprocessing",
        "//testing/pybase:parameterized",
        "//orbax/checkpoint/experimental/v1:pathways_support",
        "//orbax/checkpoint/experimental/v1/_src/handlers:pytree_handler_test_base",
    ],
)

py_test(
    name = "pytree_handler_test_multi_worker",
    srcs = ["pytree_handler_test.py"],
    args = [
        "--jax_platforms=pathways",
        "--jax_backend_target=subslice",
        "--pathways_ifrt=true",
        "--jax_allow_unused_tpus=true",
        "--pathways_expected_instances=df=1x1,df=1x1,df=1x1,df=1x1",
    ],
    main = "pytree_handler_test.py",
    deps = [
        "//pyglib/contrib/g3_multiprocessing",
        "//testing/pybase:parameterized",
        "//orbax/checkpoint/experimental/v1:pathways_support",
        "//orbax/checkpoint/experimental/v1/_src/handlers:pytree_handler_test_base",
    ],
)
