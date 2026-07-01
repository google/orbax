#ifndef THIRD_PARTY_PY_ORBAX_CHECKPOINT__SRC_SERIALIZATION_TENSORSTORE_UTILS_H_
#define THIRD_PARTY_PY_ORBAX_CHECKPOINT__SRC_SERIALIZATION_TENSORSTORE_UTILS_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "third_party/absl/status/statusor.h"
#include "third_party/json/include/nlohmann/json_fwd.hpp"

namespace orbax_checkpoint {


/*
 * Constructs a spec for a Tensorstore KvStore.
 *
 * @param directory Base path (key prefix) of the KvStore, used by the underlying
 *   file driver.
 * @param name Name (filename) of the parameter.
 * @param use_ocdbt Whether to use OCDBT driver.
 * @param process_id [only used with OCDBT driver] If provided,
 *   `{directory}/ocdbt.process_{process_id}` path is used as the base path.
 *   If a string, must conform to [A-Za-z0-9]+ pattern.
 *
 * @return A Tensorstore KvStore spec in dictionary form.
 */
absl::StatusOr<nlohmann::json>  build_kvstore_tspec(
    const std::string& directory,
    const std::optional<std::string>& name = std::nullopt,
    bool use_ocdbt = true,
    const std::optional<std::variant<int, std::string>>& process_id =
        std::nullopt);

absl::StatusOr<nlohmann::json> get_kvstore_for_gcs(
    const std::string& ckpt_path);

}  // namespace orbax_checkpoint

#endif  // THIRD_PARTY_PY_ORBAX_CHECKPOINT__SRC_SERIALIZATION_TENSORSTORE_UTILS_H_
