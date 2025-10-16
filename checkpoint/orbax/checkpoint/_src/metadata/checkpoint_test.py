# Copyright 2025 The Orbax Authors.
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

import pickle
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import root_metadata_serialization
from orbax.checkpoint._src.metadata import step_metadata_serialization


StepMetadata = checkpoint.StepMetadata
RootMetadata = checkpoint.RootMetadata
MetadataStore = checkpoint.MetadataStore
StepStatistics = step_statistics.SaveStepStatistics
CompositeItemMetadata = checkpoint.CompositeItemMetadata
SingleItemMetadata = checkpoint.SingleItemMetadata


class CheckpointMetadataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)
    self._non_blocking_store_for_writes = checkpoint.metadata_store(
        enable_write=True, blocking_write=False
    )

  def tearDown(self):
    super().tearDown()
    checkpoint._METADATA_STORE_FOR_WRITES.close()
    checkpoint._METADATA_STORE_FOR_READS.close()
    self._non_blocking_store_for_writes.close()
    checkpoint._METADATA_STORE_NON_BLOCKING_FOR_READS.close()

  def write_metadata_store(
      self, blocking_write: bool
  ) -> MetadataStore:
    if blocking_write:
      return checkpoint.metadata_store(
          enable_write=True, blocking_write=True
      )
    return self._non_blocking_store_for_writes

  def read_metadata_store(
      self, blocking_write: bool
  ) -> MetadataStore:
    return checkpoint.metadata_store(
        enable_write=False, blocking_write=blocking_write
    )

  def deserialize_metadata(
      self,
      metadata_type: type[StepMetadata] | type[RootMetadata],
      metadata: checkpoint.SerializedMetadata,
      item_metadata: CompositeItemMetadata | SingleItemMetadata | None = None,
      metrics: dict[str, Any] | None = None,
  ) -> StepMetadata | RootMetadata:
    if metadata_type == StepMetadata:
      return step_metadata_serialization.deserialize(
          metadata,
          item_metadata=item_metadata,
          metrics=metrics,
      )
    elif metadata_type == RootMetadata:
      return root_metadata_serialization.deserialize(metadata)

  def serialize_metadata(
      self,
      metadata: StepMetadata | RootMetadata,
  ) -> checkpoint.SerializedMetadata:
    if isinstance(metadata, StepMetadata):
      return step_metadata_serialization.serialize(metadata)
    elif isinstance(metadata, RootMetadata):
      return root_metadata_serialization.serialize(metadata)

  def get_metadata(
      self,
      metadata_type: type[StepMetadata] | type[RootMetadata],
      custom_metadata: dict[str, Any] = None,
  ) -> StepMetadata | RootMetadata:
    """Returns a sample metadata object of `metadata_class`."""
    if custom_metadata is None:
      custom_metadata = {'a': 1}
    if metadata_type == StepMetadata:
      return StepMetadata(
          item_handlers={'a': 'b'},
          item_metadata=None,
          metrics={'a': 1},
          performance_metrics=StepStatistics(
              step=None,
              event_type='save',
              reached_preemption=False,
              preemption_received_at=1.0,
          ),
          init_timestamp_nsecs=1,
          commit_timestamp_nsecs=1,
          custom_metadata=custom_metadata,
      )
    elif metadata_type == RootMetadata:
      return RootMetadata(
          internal_metadata={'version': 1.0},
          custom_metadata=custom_metadata,
      )

  def get_metadata_filename(
      self, metadata_type: type[StepMetadata] | type[RootMetadata]
  ) -> str:
    if metadata_type == StepMetadata:
      return checkpoint._STEP_METADATA_FILENAME
    elif metadata_type == RootMetadata:
      return checkpoint._ROOT_METADATA_FILENAME

  def get_metadata_file_path(
      self,
      metadata_type: type[StepMetadata] | type[RootMetadata],
      path: epath.PathLike = None,
  ) -> epath.Path:
    if metadata_type == StepMetadata:
      return checkpoint.step_metadata_file_path(path or self.directory)
    elif metadata_type == RootMetadata:
      return checkpoint.root_metadata_file_path(path or self.directory)

  def assertMetadataEqual(
      self, a: StepMetadata | RootMetadata, b: StepMetadata | RootMetadata,
  ):
    if isinstance(a, StepMetadata):
      self.assertEqual(a.item_handlers, b.item_handlers)
      # ignore item_metadata
      self.assertEqual(a.metrics, b.metrics)
      self.assertEqual(a.performance_metrics, b.performance_metrics)
      self.assertEqual(a.init_timestamp_nsecs, b.init_timestamp_nsecs)
      self.assertEqual(a.commit_timestamp_nsecs, b.commit_timestamp_nsecs)
      self.assertEqual(a.custom_metadata, b.custom_metadata)
    elif isinstance(a, RootMetadata):
      self.assertEqual(a.internal_metadata, b.internal_metadata)
      self.assertEqual(a.custom_metadata, b.custom_metadata)

  @parameterized.parameters(True, False)
  def test_read_unknown_path(self, blocking_write: bool):
    self.assertIsNone(
        self.write_metadata_store(blocking_write).read(
            file_path='unknown_checkpoint_path'
        )
    )

  def test_legacy_root_metadata_file_path(self):
    self.assertEqual(
        checkpoint.root_metadata_file_path(self.directory, legacy=True),
        self.directory / checkpoint._LEGACY_ROOT_METADATA_FILENAME,
    )

  @parameterized.product(
      blocking_write=[True, False],
      metadata_class=[StepMetadata, RootMetadata],
  )
  def test_write_unknown_file_path(
      self,
      blocking_write: bool,
      metadata_class: type[StepMetadata] | type[RootMetadata],
  ):
    metadata = metadata_class()

    if blocking_write:
      with self.assertRaisesRegex(ValueError, 'Metadata parent name not set'):
        self.write_metadata_store(blocking_write).write(
            file_path='unknown_metadata_path',
            metadata=self.serialize_metadata(metadata),
        )
    else:
      self.write_metadata_store(blocking_write).write(
          file_path='unknown_metadata_path',
          metadata=self.serialize_metadata(metadata),
      )
      try:
        self.write_metadata_store(blocking_write).wait_until_finished()
      except ValueError:
        # We don't want to fail the test because above write's future.result()
        # will raise 'ValueError: Metadata parent name not set...'.
        pass
      self.assertIsNone(
          self.read_metadata_store(blocking_write).read('unknown_metadata_path')
      )

  @parameterized.product(
      blocking_write=[True, False],
      metadata_class=[StepMetadata, RootMetadata],
  )
  def test_read_default_values(
      self,
      blocking_write: bool,
      metadata_class: type[StepMetadata] | type[RootMetadata],
  ):
    metadata = metadata_class()

    self.write_metadata_store(blocking_write).write(
        file_path=self.get_metadata_file_path(metadata_class),
        metadata=self.serialize_metadata(metadata),
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    serialized_metadata = self.write_metadata_store(blocking_write).read(
        file_path=self.get_metadata_file_path(metadata_class),
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        metadata,
    )

  @parameterized.product(
      blocking_write=[True, False],
      metadata_class=[StepMetadata, RootMetadata],
  )
  def test_read_with_values(
      self,
      blocking_write: bool,
      metadata_class: type[StepMetadata] | type[RootMetadata],
  ):
    metadata = self.get_metadata(metadata_class)

    self.write_metadata_store(blocking_write).write(
        file_path=self.get_metadata_file_path(metadata_class),
        metadata=self.serialize_metadata(metadata),
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    serialized_metadata = self.write_metadata_store(blocking_write).read(
        file_path=self.get_metadata_file_path(metadata_class),
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        metadata,
    )

  @parameterized.product(
      blocking_write=[True, False],
      metadata_class=[StepMetadata, RootMetadata],
  )
  def test_read_corrupt_json_data(
      self,
      blocking_write: bool,
      metadata_class: type[StepMetadata] | type[RootMetadata],
  ):
    metadata_file = self.get_metadata_file_path(metadata_class)
    metadata_file.touch()

    self.assertIsNone(
        self.write_metadata_store(blocking_write).read(
            file_path=self.get_metadata_file_path(metadata_class)
        )
    )

  @parameterized.product(
      blocking_write=[True, False],
      metadata_class=[StepMetadata, RootMetadata],
  )
  def test_update_without_prior_data(
      self,
      blocking_write: bool,
      metadata_class: type[StepMetadata] | type[RootMetadata],
  ):
    self.write_metadata_store(blocking_write).update(
        file_path=self.get_metadata_file_path(metadata_class),
        custom_metadata={'a': 1},
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    serialized_metadata = self.write_metadata_store(blocking_write).read(
        file_path=self.get_metadata_file_path(metadata_class),
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        metadata_class(
            custom_metadata={'a': 1},
        ),
    )

  @parameterized.parameters(True, False)
  def test_update_with_prior_data(
      self,
      blocking_write: bool,
  ):
    metadata = StepMetadata(init_timestamp_nsecs=1)
    self.write_metadata_store(blocking_write).write(
        file_path=self.get_metadata_file_path(StepMetadata),
        metadata=self.serialize_metadata(metadata),
    )

    self.write_metadata_store(blocking_write).update(
        file_path=self.get_metadata_file_path(StepMetadata),
        custom_metadata={'a': 1},
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    serialized_metadata = self.write_metadata_store(blocking_write).read(
        file_path=self.get_metadata_file_path(StepMetadata)
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(StepMetadata, serialized_metadata),
        StepMetadata(
            init_timestamp_nsecs=1,
            custom_metadata={'a': 1},
        ),
    )

  @parameterized.product(
      blocking_write=[True, False],
      metadata_class=[StepMetadata, RootMetadata],
  )
  def test_update_with_unknown_kwargs(
      self,
      blocking_write: bool,
      metadata_class: type[StepMetadata] | type[RootMetadata]
  ):
    metadata = metadata_class()
    self.write_metadata_store(blocking_write).write(
        file_path=self.get_metadata_file_path(metadata_class),
        metadata=self.serialize_metadata(metadata),
    )
    self.write_metadata_store(blocking_write).update(
        file_path=self.get_metadata_file_path(metadata_class),
        blah=2,
    )

    self.write_metadata_store(blocking_write).wait_until_finished()

    serialized_metadata = self.write_metadata_store(blocking_write).read(
        file_path=self.get_metadata_file_path(metadata_class)
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        metadata_class(
            custom_metadata={'blah': 2},
        ),
    )

  @parameterized.product(
      blocking_write=[True, False],
      metadata_class=[StepMetadata, RootMetadata],
  )
  def test_write_with_read_only_store_is_no_op(
      self,
      blocking_write: bool,
      metadata_class: type[StepMetadata] | type[RootMetadata],
  ):
    metadata = metadata_class()

    self.read_metadata_store(blocking_write).write(
        file_path=self.get_metadata_file_path(metadata_class),
        metadata=self.serialize_metadata(metadata),
    )

    serialized_metadata = self.read_metadata_store(blocking_write).read(
        file_path=self.get_metadata_file_path(metadata_class)
    )
    self.assertIsNone(serialized_metadata)

    serialized_metadata = self.write_metadata_store(blocking_write).read(
        file_path=self.get_metadata_file_path(metadata_class)
    )
    self.assertIsNone(serialized_metadata)

  @parameterized.parameters(StepMetadata, RootMetadata)
  def test_non_blocking_write_request_enables_writes(
      self, metadata_class: type[StepMetadata] | type[RootMetadata],
  ):
    metadata = self.get_metadata(metadata_class)

    # setup some data with blocking store.
    self.write_metadata_store(blocking_write=True).write(
        file_path=self.get_metadata_file_path(metadata_class),
        metadata=self.serialize_metadata(metadata),
    )

    serialized_metadata = self.read_metadata_store(blocking_write=True).read(
        file_path=self.get_metadata_file_path(metadata_class)
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        self.get_metadata(metadata_class),
    )

    # write validations
    serialized_metadata = self.serialize_metadata(
        self.get_metadata(metadata_class, custom_metadata={'a': 2})
    )
    self.write_metadata_store(blocking_write=False).write(
        file_path=self.get_metadata_file_path(metadata_class),
        metadata=serialized_metadata,
    )

    self.write_metadata_store(blocking_write=False).wait_until_finished()

    serialized_metadata = self.read_metadata_store(blocking_write=False).read(
        file_path=self.get_metadata_file_path(metadata_class)
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        self.get_metadata(metadata_class, custom_metadata={'a': 2}),
    )
    serialized_metadata = self.write_metadata_store(blocking_write=False).read(
        file_path=self.get_metadata_file_path(metadata_class)
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        self.get_metadata(metadata_class, custom_metadata={'a': 2}),
    )

    # update validations
    self.write_metadata_store(blocking_write=False).update(
        file_path=self.get_metadata_file_path(metadata_class),
        custom_metadata={'a': 3},
    )
    self.write_metadata_store(blocking_write=False).wait_until_finished()
    serialized_metadata = self.read_metadata_store(blocking_write=False).read(
        file_path=self.get_metadata_file_path(metadata_class)
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        self.get_metadata(metadata_class, custom_metadata={'a': 3}),
    )
    serialized_metadata = self.write_metadata_store(blocking_write=False).read(
        file_path=self.get_metadata_file_path(metadata_class)
    )
    self.assertMetadataEqual(
        self.deserialize_metadata(metadata_class, serialized_metadata),
        self.get_metadata(metadata_class, custom_metadata={'a': 3}),
    )

  @parameterized.parameters(StepMetadata, RootMetadata)
  def test_metadata_file_path(
      self, metadata_class: type[StepMetadata] | type[RootMetadata]
  ):
    self.assertEqual(
        self.get_metadata_file_path(metadata_class),
        self.directory / self.get_metadata_filename(metadata_class),
    )

    metadata_file = self.get_metadata_file_path(metadata_class)
    self.write_metadata_store(blocking_write=True).write(
        file_path=metadata_file,
        metadata=self.serialize_metadata(self.get_metadata(metadata_class)),
    )
    with self.assertRaisesRegex(IOError, 'Path is not a directory'):
      self.get_metadata_file_path(metadata_class, path=metadata_file)

  @parameterized.parameters(True, False)
  def test_pickle(self, blocking_write: bool):
    with self.assertRaisesRegex(TypeError, 'cannot pickle'):
      _ = pickle.dumps(self.write_metadata_store(blocking_write))
    _ = pickle.dumps(self.read_metadata_store(blocking_write))

  @parameterized.parameters(
      ({'internal_metadata': list()},),
      ({'internal_metadata': {int(): None}},),
      ({'custom_metadata': list()},),
      ({'custom_metadata': {int(): None}},),
  )
  def test_deserialize_wrong_types_root_metadata(
      self, wrong_metadata: checkpoint.SerializedMetadata
  ):
    with self.assertRaises(ValueError):
      self.deserialize_metadata(RootMetadata, wrong_metadata)

  @parameterized.parameters(
      ({'item_handlers': list()},),
      ({'item_handlers': {int(): None}},),
      ({'metrics': [int()]},),
      ({'metrics': {int(): None}},),
      ({'performance_metrics': list()},),
      ({'performance_metrics': {int(): float()}},),
      ({'init_timestamp_nsecs': float()},),
      ({'commit_timestamp_nsecs': float()},),
      ({'custom_metadata': list()},),
      ({'custom_metadata': {int(): None}},),
  )
  def test_deserialize_wrong_types_step_metadata(
      self, wrong_metadata: checkpoint.SerializedMetadata
  ):
    with self.assertRaises(ValueError):
      self.deserialize_metadata(StepMetadata, wrong_metadata)

  @parameterized.parameters(StepMetadata, RootMetadata)
  def test_unknown_key_in_metadata(
      self, metadata_class: type[StepMetadata] | type[RootMetadata],
  ):
    metadata = metadata_class(
        custom_metadata={'a': 1},
    )
    serialized_metadata = self.serialize_metadata(metadata)
    serialized_metadata['blah'] = 2
    with self.assertRaisesRegex(
        ValueError, 'Provided metadata contains unknown key blah'
    ):
      self.deserialize_metadata(metadata_class, serialized_metadata)

  @parameterized.named_parameters(
      ('single', 'a_handler'),
      ('composite', {'a': 'a_handler'}),
  )
  def test_deserialize_item_handlers(self, item_handlers):
    serialized_metadata = {'item_handlers': item_handlers}
    metadata = self.deserialize_metadata(StepMetadata, serialized_metadata)
    self.assertEqual(metadata.item_handlers, item_handlers)

  @parameterized.named_parameters(
      ('single', 'a'),
      ('composite', {'A': 'a'}),
  )
  def test_deserialize_item_metadata_with_item_metadata_kwarg(
      self, item_metadata
  ):
    metadata = self.deserialize_metadata(
        StepMetadata, {}, item_metadata=item_metadata,
    )
    self.assertEqual(metadata.item_metadata, item_metadata)

  @parameterized.parameters(
      ('metrics', 1, dict, int),
  )
  def test_deserialize_metrics_kwarg(
      self, kwarg_name, kwarg_value, expected_type, wrong_type
  ):
    with self.assertRaisesRegex(
        ValueError,
        f'Object must be of type {expected_type}, got {wrong_type}.',
    ):
      self.deserialize_metadata(StepMetadata, {}, **{kwarg_name: kwarg_value})

  @parameterized.parameters(
      ({'metrics': 1}, dict, int),
      ({'init_timestamp_nsecs': 'a'}, int, str),
      ({'commit_timestamp_nsecs': 'a'}, int, str),
      ({'custom_metadata': 1}, dict, int),
  )
  def test_validate_field_wrong_type(
      self, step_metadata, expected_field_type, wrong_field_type
  ):
    with self.assertRaisesRegex(
        ValueError,
        f'Object must be of type {expected_field_type}, got {wrong_field_type}',
    ):
      self.deserialize_metadata(StepMetadata, step_metadata)

  @parameterized.parameters(
      ({'item_handlers': 1}, int),
      ({'performance_metrics': 1}, int),
  )
  def test_validate_field_sequence_wrong_type(
      self, step_metadata, wrong_field_type
  ):
    with self.assertRaisesRegex(
        ValueError,
        f'Object must be any one of types '
        r'\[.+]'
        f', got {wrong_field_type}.',
    ):
      self.deserialize_metadata(StepMetadata, step_metadata)

  @parameterized.parameters(
      ({'item_handlers': {1: 'a'}}, str, int),
      ({'metrics': {1: 'a'}}, str, int),
      ({'performance_metrics': {1: 1.0}}, str, int),
      ({'custom_metadata': {1: 'a'}}, str, int),
  )
  def test_validate_dict_entry_wrong_key_type(
      self, step_metadata, expected_key_type, wrong_key_type
  ):
    with self.assertRaisesRegex(
        ValueError,
        f'Object must be of type {expected_key_type}, got {wrong_key_type}.',
    ):
      self.deserialize_metadata(StepMetadata, step_metadata)

  @parameterized.parameters(
      ({'item_handlers': {'a': 'b'}},),
      ({'performance_metrics': {'a': 1.0}},),
      ({'custom_metadata': {'a': 1}, 'init_timestamp_nsecs': 1},),
  )
  def test_serialize_for_update_valid_kwargs(
      self, kwargs: dict[str, Any]
  ):
    expected_kwargs = kwargs.copy()
    if 'custom_metadata' in expected_kwargs:
      expected_kwargs['custom_metadata'] = expected_kwargs.pop(
          'custom_metadata'
      )
    self.assertEqual(
        step_metadata_serialization.serialize_for_update(**kwargs),
        expected_kwargs,
    )

  @parameterized.parameters(
      ({'item_handlers': list()},),
      ({'item_handlers': {int(): None}},),
      ({'metrics': [int()]},),
      ({'metrics': {int(): None}},),
      ({'performance_metrics': list()},),
      ({'init_timestamp_nsecs': float()},),
      ({'commit_timestamp_nsecs': float()},),
      ({'custom_metadata': list()},),
      ({'custom_metadata': {int(): None}},),
  )
  def test_serialize_for_update_wrong_types(
      self, kwargs: dict[str, Any]
  ):
    with self.assertRaises(ValueError):
      step_metadata_serialization.serialize_for_update(**kwargs)

  def test_serialize_for_update_with_unknown_kwargs(self):
    with self.assertRaisesRegex(
        ValueError, 'Provided metadata contains unknown key blah'
    ):
      step_metadata_serialization.serialize_for_update(
          custom_metadata={'a': 1},
          blah=123,
      )

    with self.assertRaisesRegex(
        ValueError, 'Provided metadata contains unknown key custom'
    ):
      step_metadata_serialization.serialize_for_update(
          custom={'a': 1},
      )

  def test_serialize_for_update_performance_metrics_only_float(self):
    self.assertEqual(
        step_metadata_serialization.serialize_for_update(
            performance_metrics=StepStatistics(
                step=1,
                event_type='save',
                reached_preemption=False,
                preemption_received_at=1.0,
            )
        ),
        {'performance_metrics': {'preemption_received_at': 1.0}},
    )


if __name__ == '__main__':
  absltest.main()
