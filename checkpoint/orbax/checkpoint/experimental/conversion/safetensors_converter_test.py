# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Safetensors <-> Orbax checkpoint conversion."""

from __future__ import annotations

import asyncio
import json
import os
import struct
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp_v0
from orbax.checkpoint.experimental import conversion
from orbax.checkpoint.experimental import v1 as ocp
from orbax.checkpoint.experimental.mcp import format_inspector
import safetensors.numpy


def _write_raw_safetensors(
    path: str,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    metadata: dict[str, str] | None = None,
) -> None:
  """Writes a Safetensors file directly from raw `(st_dtype, shape, bytes)`."""
  header: dict[str, Any] = {}
  if metadata:
    header["__metadata__"] = metadata
  offset = 0
  chunks: list[bytes] = []
  for name, (st_dtype, shape, payload) in tensors.items():
    header[name] = {
        "dtype": st_dtype,
        "shape": list(shape),
        "data_offsets": [offset, offset + len(payload)],
    }
    offset += len(payload)
    chunks.append(payload)
  raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
  pad = (8 - (len(raw_header) % 8)) % 8
  padded_header = raw_header + (b" " * pad)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(padded_header)))
    f.write(padded_header)
    for chunk in chunks:
      f.write(chunk)


def _save_v1(
    path: str, state: Any, *, custom_metadata: Any = None
) -> None:
  """Typed wrapper around `ocp.save` for test trees."""
  ocp.save(path, state, custom_metadata=custom_metadata)  # pyrefly: ignore[bad-argument-type]


def _assert_trees_exact_equal(
    test_case: absltest.TestCase, actual: Any, expected: Any
) -> None:
  """Asserts identical PyTree structure, dtypes, shapes, and exact bit values."""
  test_case.assertEqual(
      jax.tree.structure(actual), jax.tree.structure(expected)
  )

  def _check_leaf(a: Any, b: Any) -> None:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    test_case.assertEqual(arr_a.dtype, arr_b.dtype)
    test_case.assertEqual(arr_a.shape, arr_b.shape)
    np.testing.assert_array_equal(
        arr_a.reshape(-1).view(np.uint8),
        arr_b.reshape(-1).view(np.uint8),
    )

  jax.tree.map(_check_leaf, actual, expected)


class SafetensorsConverterTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.root = self.create_tempdir().full_path

  @parameterized.product(
      use_ocdbt=[True, False],
      use_zarr3=[False, True],
      use_compression=[True, False],
  )
  def test_nested_tree_round_trip_both_directions(
      self,
      use_ocdbt: bool,
      use_zarr3: bool,
      use_compression: bool,
  ) -> None:
    nested_tree = {
        "embed": {
            "tokens": np.arange(32 * 16, dtype=np.float32).reshape(32, 16),
        },
        "layers": {
            "0": {
                "attn": {
                    "q": np.linspace(-1.0, 1.0, 64, dtype=jnp.bfloat16).reshape(
                        8, 8
                    ),
                    "k": np.arange(64, dtype=np.float16).reshape(8, 8),
                },
                "norm_scale": np.ones((8,), dtype=np.float64),
            },
            "1": {
                "gate": np.array(
                    [[True, False], [False, True]], dtype=np.bool_
                ),
                "counts": np.array([-5, 0, 7, 42], dtype=np.int64),
            },
        },
        "step": np.array(1234, dtype=np.int32),
    }
    custom_meta = {
        "model_family": "test-transformer",
        "step": 1234,
        "hparams": {"hidden_size": 16, "num_layers": 2},
    }

    orbax_src = os.path.join(self.root, "orbax_src")
    ctx = ocp.Context()
    ctx.array.saving.use_ocdbt = use_ocdbt
    ctx.array.saving.use_zarr3 = use_zarr3
    ctx.array.saving.use_compression = use_compression
    with ctx:
      _save_v1(orbax_src, nested_tree, custom_metadata=custom_meta)

    st_dir = os.path.join(self.root, "st_mid")
    opts = conversion.ConversionOptions(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        use_compression=use_compression,
    )
    res_to_st = conversion.orbax_to_safetensors(orbax_src, st_dir, options=opts)
    self.assertEqual(res_to_st.dest_format, "safetensors")
    self.assertEqual(res_to_st.num_tensors, 7)
    self.assertEqual(res_to_st.num_output_shards, 1)
    self.assertTrue(os.path.isfile(os.path.join(st_dir, "model.safetensors")))

    # Verify Orbax SafetensorsLayout can load the written Safetensors file.
    with ocp.Context(
        checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS
    ):
      loaded_st: Any = ocp.load(st_dir)
    self.assertIn("layers.0.attn.q", loaded_st)
    np.testing.assert_array_equal(
        np.asarray(loaded_st["layers.0.attn.q"]).view(np.uint16),
        nested_tree["layers"]["0"]["attn"]["q"].view(np.uint16),
    )

    # Convert back to Orbax and verify exact nested PyTree & metadata match.
    orbax_rt = os.path.join(self.root, "orbax_rt")
    res_to_orbax = conversion.safetensors_to_orbax(
        st_dir, orbax_rt, options=opts
    )
    self.assertEqual(res_to_orbax.num_tensors, 7)
    self.assertEqual(
        res_to_orbax.dest_format,
        "orbax-ocdbt" if use_ocdbt else "orbax-zarr",
    )

    restored_tree = ocp.load(orbax_rt)
    _assert_trees_exact_equal(self, restored_tree, nested_tree)
    restored_meta = ocp.checkpointables_metadata(orbax_rt)
    self.assertEqual(restored_meta.custom_metadata, custom_meta)

    # Also verify sharded jax.Array restoration via abstract_state (global
    # jax_enable_x64 is required because ocp.load runs in a background thread).
    jax.config.update("jax_enable_x64", True)
    try:
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
      rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      abstract_state = jax.tree.map(
          lambda x: jax.ShapeDtypeStruct(
              shape=x.shape, dtype=x.dtype, sharding=rep
          ),
          nested_tree,
      )
      restored_jax: Any = ocp.load(orbax_rt, abstract_state=abstract_state)
      self.assertIsInstance(restored_jax["embed"]["tokens"], jax.Array)
      _assert_trees_exact_equal(self, restored_jax, nested_tree)
    finally:
      jax.config.update("jax_enable_x64", False)

    # Verify MCP format_inspector recognizes converted Orbax checkpoint.
    info = asyncio.run(format_inspector.inspect_format(orbax_rt))
    self.assertTrue(info.is_orbax)
    self.assertEqual(info.checkpoint_type, "Orbax V1")
    tensor_meta = asyncio.run(
        format_inspector.inspect_tensor_metadata(orbax_rt)
    )
    self.assertEqual(tensor_meta["total_matched_tensors"], 7)

  def test_all_supported_dtypes_round_trip(self) -> None:
    dtypes_and_arrays: dict[str, np.ndarray] = {
        "bool": np.array([True, False, True], dtype=np.bool_),
        "i8": np.array([-128, 0, 127], dtype=np.int8),
        "u8": np.array([0, 128, 255], dtype=np.uint8),
        "i16": np.array([-30000, 0, 30000], dtype=np.int16),
        "u16": np.array([0, 1000, 65535], dtype=np.uint16),
        "i32": np.array([-100000, 0, 100000], dtype=np.int32),
        "u32": np.array([0, 42, 4000000000], dtype=np.uint32),
        "i64": np.array([-2**40, 0, 2**40], dtype=np.int64),
        "u64": np.array([0, 1, 2**40], dtype=np.uint64),
        "f16": np.array([-1.5, 0.0, 3.25], dtype=np.float16),
        "f32": np.array([-1e-5, 0.0, 3.1415927], dtype=np.float32),
        "f64": np.array([-1e-12, 0.0, 2.718281828459045], dtype=np.float64),
        "bf16": np.array([-2.5, 0.0, 7.5], dtype=jnp.bfloat16),
        "f8_e4m3": np.array([-1.0, 0.0, 2.0], dtype=jnp.float8_e4m3fn),
        "f8_e5m2": np.array([-1.0, 0.0, 4.0], dtype=jnp.float8_e5m2),
        "c64": np.array([1.5 - 2.5j, -0.25 + 4.0j], dtype=np.complex64),
    }
    orbax_dir = os.path.join(self.root, "dtypes_orbax")
    _save_v1(orbax_dir, dtypes_and_arrays)

    st_file = os.path.join(self.root, "dtypes.safetensors")
    conversion.convert_checkpoint(orbax_dir, st_file)

    orbax_rt = os.path.join(self.root, "dtypes_orbax_rt")
    conversion.convert_checkpoint(st_file, orbax_rt)

    restored = ocp.load(orbax_rt)
    _assert_trees_exact_equal(self, restored, dtypes_and_arrays)

  def test_multi_file_sharded_safetensors_round_trip(self) -> None:
    # Build an Orbax checkpoint totaling ~48 KiB across 6 tensors (8 KiB each),
    # then shard into 20 KiB max_shard_size_bytes -> 3 shards + index.json.
    tree = {
        "model": {
            f"layer_{i}": {
                "weight": (
                    np.arange(2048, dtype=np.float32).reshape(64, 32)
                    + i * 100.0
                )
            }
            for i in range(6)
        }
    }
    orbax_src = os.path.join(self.root, "multi_orbax_src")
    _save_v1(orbax_src, tree, custom_metadata={"format": "jax"})

    sharded_st_dir = os.path.join(self.root, "sharded_st")
    opts = conversion.ConversionOptions(max_shard_size_bytes=20 * 1024)
    res_st = conversion.orbax_to_safetensors(
        orbax_src, sharded_st_dir, options=opts
    )
    self.assertEqual(res_st.num_output_shards, 3)
    self.assertEqual(res_st.num_tensors, 6)

    expected_shards = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    for fname in expected_shards:
      shard_path = os.path.join(sharded_st_dir, fname)
      self.assertTrue(os.path.isfile(shard_path))
      # Verify third-party safetensors reader also parses each shard cleanly.
      loaded_part = safetensors.numpy.load_file(shard_path)
      self.assertLen(loaded_part, 2)

    index_path = os.path.join(sharded_st_dir, "model.safetensors.index.json")
    self.assertTrue(os.path.isfile(index_path))
    index_json = json.loads(epath.Path(index_path).read_text())
    self.assertEqual(index_json["metadata"]["total_size"], 6 * 2048 * 4)
    self.assertLen(index_json["weight_map"], 6)

    # Convert multi-shard Safetensors back to Orbax and verify exact match.
    orbax_rt = os.path.join(self.root, "multi_orbax_rt")
    res_orbax = conversion.safetensors_to_orbax(sharded_st_dir, orbax_rt)
    self.assertEqual(res_orbax.num_tensors, 6)
    _assert_trees_exact_equal(self, ocp.load(orbax_rt), tree)

    # Convert back to single-file Safetensors and compare shard contents.
    single_st_dir = os.path.join(self.root, "single_st_rt")
    res_single = conversion.orbax_to_safetensors(orbax_rt, single_st_dir)
    self.assertEqual(res_single.num_output_shards, 1)
    merged_loaded = safetensors.numpy.load_file(
        os.path.join(single_st_dir, "model.safetensors")
    )
    for i in range(6):
      np.testing.assert_array_equal(
          merged_loaded[f"model.layer_{i}.weight"],
          tree["model"][f"layer_{i}"]["weight"],
      )

    # Deleting a referenced shard must fail fast via index.json validation.
    os.remove(os.path.join(sharded_st_dir, "model-00002-of-00003.safetensors"))
    with self.assertRaisesRegex(FileNotFoundError, "missing shard"):
      conversion.safetensors_to_orbax(
          sharded_st_dir, os.path.join(self.root, "broken_dst")
      )

  @parameterized.product(use_zarr3=[False, True])
  def test_streaming_chunks_when_tensors_exceed_memory_budget(
      self, use_zarr3: bool
  ) -> None:
    # Each tensor here is 64 KiB .. 96 KiB, while max_in_flight_bytes is set to
    # just 4 KiB (and includes a 3D tensor whose single outer slice > 4 KiB so
    # inner-axis slicing is also exercised).
    st_src_dir = os.path.join(self.root, f"large_st_{use_zarr3}")
    os.makedirs(st_src_dir, exist_ok=True)
    tensors = {
        "emb.weight": np.arange(16384, dtype=np.float32).reshape(256, 64),
        "wide.row": np.arange(6144, dtype=np.float32).reshape(2, 3072),
        "small.bias": np.arange(32, dtype=np.float32),
    }
    safetensors.numpy.save_file(
        tensors, os.path.join(st_src_dir, "model.safetensors")
    )

    tiny_budget_opts = conversion.ConversionOptions(
        max_in_flight_bytes=4096,
        use_ocdbt=True,
        use_zarr3=use_zarr3,
    )
    orbax_mid = os.path.join(self.root, f"large_orbax_{use_zarr3}")
    res1 = conversion.safetensors_to_orbax(
        st_src_dir, orbax_mid, options=tiny_budget_opts
    )
    self.assertEqual(res1.num_tensors, 3)

    st_rt_dir = os.path.join(self.root, f"large_st_rt_{use_zarr3}")
    res2 = conversion.orbax_to_safetensors(
        orbax_mid, st_rt_dir, options=tiny_budget_opts
    )
    self.assertEqual(res2.num_tensors, 3)

    rt_tensors = safetensors.numpy.load_file(
        os.path.join(st_rt_dir, "model.safetensors")
    )
    for k, expected in tensors.items():
      np.testing.assert_array_equal(rt_tensors[k], expected)

  def test_prefix_collision_fallback_and_flat_keys_mode(self) -> None:
    colliding = {
        "adapter": np.array([1.0, 2.0], dtype=np.float32),
        "adapter.weight": np.array([[3.0, 4.0]], dtype=np.float32),
    }
    st_file = os.path.join(self.root, "colliding.safetensors")
    safetensors.numpy.save_file(colliding, st_file)

    # strict_unflatten=True raises on prefix collision.
    with self.assertRaisesRegex(ValueError, "collisions"):
      conversion.safetensors_to_orbax(
          st_file,
          os.path.join(self.root, "strict_fail"),
          options=conversion.ConversionOptions(strict_unflatten=True),
      )

    # Default strict_unflatten=False falls back cleanly to flat dict + warning.
    orbax_flat = os.path.join(self.root, "fallback_flat")
    res = conversion.safetensors_to_orbax(st_file, orbax_flat)
    self.assertNotEmpty(res.metadata_warnings)
    restored: Any = ocp.load(orbax_flat)
    self.assertCountEqual(restored.keys(), ["adapter", "adapter.weight"])
    _assert_trees_exact_equal(self, restored, colliding)

    # Round-trip flat Orbax dict back to Safetensors with key_separator=None.
    st_rt = os.path.join(self.root, "colliding_rt.safetensors")
    conversion.orbax_to_safetensors(
        orbax_flat,
        st_rt,
        options=conversion.ConversionOptions(key_separator=None),
    )
    _assert_trees_exact_equal(
        self, safetensors.numpy.load_file(st_rt), colliding
    )

  def test_orbax_v0_flat_and_composite_checkpoints(self) -> None:
    v0_flat_dir = os.path.join(self.root, "v0_flat")
    tree = {"a": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)}}
    with ocp_v0.Checkpointer(ocp_v0.StandardCheckpointHandler()) as ckptr:
      ckptr.save(epath.Path(v0_flat_dir), args=ocp_v0.args.StandardSave(tree))

    st_from_v0 = os.path.join(self.root, "st_from_v0")
    res = conversion.convert_checkpoint(v0_flat_dir, st_from_v0)
    self.assertEqual(res.source_format, "orbax-ocdbt")
    self.assertEqual(res.num_tensors, 1)
    np.testing.assert_array_equal(
        safetensors.numpy.load_file(
            os.path.join(st_from_v0, "model.safetensors")
        )["a.w"],
        tree["a"]["w"],
    )

    # Multi-item V1 checkpoint converted with checkpointable_name=None.
    v1_multi = os.path.join(self.root, "v1_multi")
    items = {
        "encoder": {"k": jnp.ones((2, 2), dtype=jnp.float32)},
        "decoder": {"b": jnp.zeros((2,), dtype=jnp.float32)},
    }
    ocp.save_checkpointables(v1_multi, items)
    st_all = os.path.join(self.root, "st_all")
    conversion.orbax_to_safetensors(
        v1_multi,
        st_all,
        options=conversion.ConversionOptions(checkpointable_name=None),
    )
    loaded_all = safetensors.numpy.load_file(
        os.path.join(st_all, "model.safetensors")
    )
    self.assertCountEqual(loaded_all.keys(), ["encoder.k", "decoder.b"])

  def test_unsupported_dtypes_and_non_tensor_leaves(self) -> None:
    # 1. Orbax complex128 -> Safetensors raises UnsupportedDtypeError.
    c128_dir = os.path.join(self.root, "c128_orbax")
    _save_v1(c128_dir, {"z": np.array([1.0 + 2.0j], dtype=np.complex128)})
    with self.assertRaises(conversion.UnsupportedDtypeError):
      conversion.orbax_to_safetensors(
          c128_dir, os.path.join(self.root, "c128_st")
      )

    # 2. Safetensors F4 -> Orbax raises UnsupportedDtypeError before writing.
    f4_file = os.path.join(self.root, "packed_f4.safetensors")
    _write_raw_safetensors(f4_file, {"x": ("F4", (4,), b"\x12\x34")})
    f4_dst = os.path.join(self.root, "f4_orbax")
    with self.assertRaises(conversion.UnsupportedDtypeError):
      conversion.safetensors_to_orbax(f4_file, f4_dst)
    self.assertFalse(os.path.exists(f4_dst))

    # 3. String leaves fail by default and are recorded when dropped.
    mixed_dir = os.path.join(self.root, "mixed_orbax")
    _save_v1(
        mixed_dir,
        {"w": np.array([1.0, 2.0], dtype=np.float32), "note": "hello"},
    )
    with self.assertRaisesRegex(ValueError, "non-tensor"):
      conversion.orbax_to_safetensors(
          mixed_dir, os.path.join(self.root, "mixed_fail")
      )
    res_drop = conversion.orbax_to_safetensors(
        mixed_dir,
        os.path.join(self.root, "mixed_ok"),
        options=conversion.ConversionOptions(drop_non_tensor_leaves=True),
    )
    self.assertEqual(res_drop.dropped_non_tensor_leaves, ("note",))
    self.assertEqual(res_drop.num_tensors, 1)

  def test_overwrite_and_same_path_guards(self) -> None:
    st_file = os.path.join(self.root, "src.safetensors")
    safetensors.numpy.save_file({"a": np.ones((4,), dtype=np.float32)}, st_file)

    with self.assertRaisesRegex(ValueError, "must be different"):
      conversion.convert_checkpoint(
          st_file,
          st_file,
          options=conversion.ConversionOptions(overwrite=True),
      )

    dst_dir = os.path.join(self.root, "dst_orbax")
    conversion.convert_checkpoint(st_file, dst_dir)
    with self.assertRaises(FileExistsError):
      conversion.convert_checkpoint(st_file, dst_dir)
    res = conversion.convert_checkpoint(
        st_file,
        dst_dir,
        options=conversion.ConversionOptions(overwrite=True),
    )
    self.assertEqual(res.num_tensors, 1)


if __name__ == "__main__":
  absltest.main()
