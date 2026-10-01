#include "third_party/py/orbax/checkpoint/_src/serialization/tensorstore_utils.h"

#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <regex>  // NOLINT(build/c++11)
#include <string>
#include <utility>
#include <variant>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/json/include/nlohmann/json.hpp"
#include "third_party/json/include/nlohmann/json_fwd.hpp"

namespace orbax_checkpoint {

namespace {

const char kDefaultDriver[] = "file";
const char kProcessSubdirPrefix[] = "ocdbt.process_";
const char kOcdbtProcessIdRe[] = "[A-Za-z0-9]+";
const char kGcsPathRe[] = "^gs://([^/]*)/(.*)$";

}  // namespace

absl::StatusOr<nlohmann::json> get_kvstore_for_gcs(
    const std::string& ckpt_path) {
  std::regex gcs_path_regex(kGcsPathRe);
  std::smatch match;
  if (!std::regex_match(ckpt_path, match, gcs_path_regex)) {
    return absl::InvalidArgumentError(
        "The ckpt_path should contain the bucket name and the "
        "file path inside the bucket. Got: " +
        ckpt_path);
  }
  std::string gcs_bucket = match[1];
  std::string path_without_bucket = match[2];
  nlohmann::json json_spec = {
      {"driver", "gcs"}, {"bucket", gcs_bucket}, {"path", path_without_bucket}};
  return absl::StatusOr<nlohmann::json>(json_spec);
}

absl::StatusOr<nlohmann::json> build_kvstore_tspec(
    const std::string& directory, const std::optional<std::string>& name,
    bool use_ocdbt,
    const std::optional<std::variant<int, std::string>>& process_id) {
  std::string default_driver = std::string(kDefaultDriver);

  std::string normalized_directory =
      std::filesystem::path(directory).lexically_normal().string();
  normalized_directory =
      std::regex_replace(normalized_directory, std::regex("gs:/"), "gs://");

  bool is_gcs_path = normalized_directory.starts_with("gs://");

  nlohmann::json kv_spec;

  if (use_ocdbt) {
    if (!is_gcs_path &&
        !std::filesystem::path(normalized_directory).is_absolute()) {
      return absl::InvalidArgumentError(
          "Checkpoint path should be absolute. Got " + normalized_directory);
    }

    if (process_id.has_value()) {
      std::string process_id_str;
      if (std::holds_alternative<int>(*process_id)) {
        process_id_str = std::to_string(std::get<int>(*process_id));
      } else {
        process_id_str = std::get<std::string>(*process_id);
      }
      std::regex process_id_regex(kOcdbtProcessIdRe);
      if (!std::regex_match(process_id_str, process_id_regex)) {
        return absl::InvalidArgumentError("process_id must conform to " +
                                          std::string(kOcdbtProcessIdRe) +
                                          " pattern, got " + process_id_str);
      }
      normalized_directory =
          (std::filesystem::path(normalized_directory) /
           (std::string(kProcessSubdirPrefix) + process_id_str))
              .string();
    }

    nlohmann::json base_driver_spec;
    if (is_gcs_path) {
      base_driver_spec = normalized_directory;
    } else {
      base_driver_spec = nlohmann::json {
        {"driver", default_driver}, { "path", normalized_directory }
      };
    }

    kv_spec["driver"] = "ocdbt";
    kv_spec["base"] = base_driver_spec;

    if (name.has_value()) {
      kv_spec["path"] = *name;
    }

    kv_spec["experimental_read_coalescing_threshold_bytes"] = 1000000;
    kv_spec["experimental_read_coalescing_merged_bytes"] = 500000000000;
    kv_spec["experimental_read_coalescing_interval"] = "1ms";
    kv_spec["cache_pool"] = "cache_pool#ocdbt";

  } else {
    std::string path =
        name.has_value()
            ? (std::filesystem::path(normalized_directory) / *name).string()
            : normalized_directory;

    if (is_gcs_path) {
      absl::StatusOr<nlohmann::json> gcs_kvstore = get_kvstore_for_gcs(path);
      if (!gcs_kvstore.ok()) {
        return gcs_kvstore.status();
      }
      kv_spec = std::move(gcs_kvstore).value();
    } else {
      kv_spec["driver"] = default_driver;
      kv_spec["path"] = path;
    }
  }

  return kv_spec;
}

}  // namespace orbax_checkpoint
