#include "third_party/py/orbax/checkpoint/_src/serialization/tensorstore_utils.h"

#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/json/include/nlohmann/json.hpp"
#include "third_party/json/include/nlohmann/json_fwd.hpp"

namespace orbax_checkpoint {
namespace {

using ::testing::TestWithParam;
using ::testing::status::StatusIs;

struct FormattingTestCase {
  std::string test_name;
  std::string directory;
  std::string param_name;
  bool use_ocdbt;
  int process_id;
  std::string expected_tspec_json_str;
};

using TensorStoreUtilTest = TestWithParam<FormattingTestCase>;

TEST_P(TensorStoreUtilTest, BuildKvstoreTspec) {
  const FormattingTestCase& test_case = GetParam();
  ASSERT_OK_AND_ASSIGN(
      nlohmann::json json_kvstore_spec,
      build_kvstore_tspec(test_case.directory, test_case.param_name,
                          test_case.use_ocdbt, test_case.process_id));
  nlohmann::json expected_json_kvstore_spec =
      nlohmann::json::parse(test_case.expected_tspec_json_str);
  EXPECT_TRUE(json_kvstore_spec == expected_json_kvstore_spec);
}

INSTANTIATE_TEST_SUITE_P(
    NumbersTestSuiteInstantiation, TensorStoreUtilTest,
    testing::ValuesIn<FormattingTestCase>({
        {"local_fs_path", "/tmp/local_path", "params/a", false, 13,
         R"({"driver":"gfile","path":"/tmp/local_path/params/a"})"},
        {"regular_gcs_path_with_ocdbt", "gs://gcs_bucket/object_path",
         "params/a", true, 0,
         R"({"driver": "ocdbt",
         "base": "gs://gcs_bucket/object_path/ocdbt.process_0",
         "path": "params/a",
         "experimental_read_coalescing_threshold_bytes": 1000000,
         "experimental_read_coalescing_merged_bytes": 500000000000,
         "experimental_read_coalescing_interval": "1ms",
         "cache_pool": "cache_pool#ocdbt"})"},
    }),
    [](const testing::TestParamInfo<TensorStoreUtilTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(TensorStoreUtilSimpleTest, GetKvstoreForGcs) {
  std::vector<std::string> valid_paths = {
      "gs://my-bucket/data/file.txt",
      "gs://another-bucket/folder/",
  };

  std::vector<nlohmann::json> expected_json{
      {{"bucket", "my-bucket"}, {"driver", "gcs"}, {"path", "data/file.txt"}},
      {{"bucket", "another-bucket"}, {"driver", "gcs"}, {"path", "folder/"}}};

  for (int i = 0; i < valid_paths.size(); i++) {
    ASSERT_OK_AND_ASSIGN(auto json_spec, get_kvstore_for_gcs(valid_paths[i]));
    ASSERT_EQ(json_spec, expected_json[i]);
  }

  std::vector<std::string> invalid_paths = {"gs://invalid-path",
                                            "https://www.example.com"};
  for (int i = 0; i < invalid_paths.size(); i++) {
    EXPECT_THAT(get_kvstore_for_gcs(invalid_paths[i]),
                StatusIs(util::error::INVALID_ARGUMENT));
  }
}

}  // namespace
}  // namespace orbax_checkpoint
