package(default_visibility = ["//visibility:public"])

py_library(
    name = "checkpoint_manager",
    srcs = ["checkpoint_manager.py"],
    deps = [
        ":local_checkpoint_data_debugging",
        ":mesh_consistency",
        "//checkpoint/orbax/checkpoint:abstract_checkpoint_manager",
        "//checkpoint/orbax/checkpoint:args",
        "//checkpoint/orbax/checkpoint:checkpoint_manager",
        "//checkpoint/orbax/checkpoint:checkpoint_utils",
        "//checkpoint/orbax/checkpoint:logging",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/multihost:multislice",
        "//checkpoint/orbax/checkpoint/_src/path:step",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        ":checkpoint_manager",
        ":mesh_consistency",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint:path",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint:utils",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/multihost:multislice",
    ],
)

py_library(
    name = "multihost",
    srcs = ["multihost.py"],
    deps = ["//checkpoint/orbax/checkpoint/_src/multihost"],
)

py_library(
    name = "replicator_checkpoint_manager",
    srcs = ["replicator_checkpoint_manager.py"],
    deps = [
        ":mesh_consistency",
        "//checkpoint/orbax/checkpoint:abstract_checkpoint_manager",
        "//checkpoint/orbax/checkpoint:args",
        "//checkpoint/orbax/checkpoint:checkpoint_manager",
        "//checkpoint/orbax/checkpoint:path",
        "//checkpoint/orbax/checkpoint/_src:composite",
        "//checkpoint/orbax/checkpoint/_src/handlers:handler_registration",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
    ],
)

py_library(
    name = "mesh_consistency",
    srcs = ["mesh_consistency.py"],
    deps = [
        ":multihost",
        "//checkpoint/orbax/checkpoint:options",
        "//checkpoint/orbax/checkpoint:path",
    ],
)

py_library(
    name = "local_checkpoint_data_debugging",
    srcs = ["local_checkpoint_data_debugging.py"],
    deps = [
        "//checkpoint/orbax/checkpoint/_src/arrays:abstract_arrays",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
        "//checkpoint/orbax/checkpoint/_src/serialization:types",
        "//checkpoint/orbax/checkpoint/_src/tree:utils",
    ],
)

py_library(
    name = "integration_test_utils",
    srcs = ["integration_test_utils.py"],
    deps = [
        ":checkpoint_manager",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/multihost:multislice",
    ],
)

py_test(
    name = "checkpoint_manager_test",
    timeout = "long",
    srcs = ["checkpoint_manager_test.py"],
    args = [
        "--alsologtostderr",
        "--logbuflevel=-1",  # Ensures logs are saved even if the test times out.
        "--extra_test_args='--vmodule=checkpoint_manager=1'",
    ],
    deps = [":test_utils"],
)

py_test(
    name = "local_checkpoint_manager_test",
    srcs = ["local_checkpoint_manager_test.py"],
    args = [
        "--alsologtostderr",
        "--logbuflevel=-1",  # Ensures logs are saved even if the test times out.
        "--extra_test_args='--vmodule=checkpoint_manager=1'",
    ],
    deps = [":test_utils"],
)

py_test(
    name = "replicator_checkpoint_manager_test",
    srcs = ["replicator_checkpoint_manager_test.py"],
    args = [
        "--alsologtostderr",
        "--logbuflevel=-1",  # Ensures logs are saved even if the test times out.
    ],
    deps = [
        ":mesh_consistency",
        ":replicator_checkpoint_manager",
        ":test_utils",
        "//checkpoint/orbax/checkpoint:args",
        "//checkpoint/orbax/checkpoint:path",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/handlers:pytree_checkpoint_handler",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/multihost:multislice",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
    ],
)

py_test(
    name = "local_checkpoint_data_debugging_test",
    srcs = ["local_checkpoint_data_debugging_test.py"],
    args = [
        "--alsologtostderr",
        "--logbuflevel=-1",  # Ensures logs are saved even if the test times out.
    ],
    deps = [
        ":local_checkpoint_data_debugging",
        "//checkpoint/orbax/checkpoint:test_utils",
        "//checkpoint/orbax/checkpoint/_src/arrays:types",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/serialization",
        "//checkpoint/orbax/checkpoint/_src/serialization:tensorstore_utils",
        "//checkpoint/orbax/checkpoint/_src/serialization:type_handlers",
    ],
)

orbax_multiprocess_test(
    name = "multihost_test",
    size = "small",
    srcs = ["multihost_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    deps = [
        ":multihost",
        "//checkpoint/orbax/checkpoint/_src/multihost",
    ],
)

py_test(
    name = "multihost_simulated_test",
    srcs = ["multihost_simulated_test.py"],
    deps = [":multihost"],
)

py_binary(
    name = "broadcast_multislice_test_binary",
    testonly = 1,
    srcs = ["broadcast_multislice_test.py"],
    main = "broadcast_multislice_test.py",
    deps = [
        ":integration_test_utils",
        "//checkpoint/orbax/checkpoint:utils",
        "//checkpoint/orbax/checkpoint/_src/multihost",
        "//checkpoint/orbax/checkpoint/_src/multihost:multislice",
    ],
)

megascale_multi_slice_test(
    name = "broadcast_multislice_test",
    num_slices = 4,
    test_binary = ":broadcast_multislice_test_binary",
    use_jax_service = True,
)
