#include "inference_backend.h"

#include "logger_manager.h"

#include <utility>

bool InferenceBackend::degradeToSyncInference(void * input_data, std::vector<void *> output_data) {
    const std::string reason = asyncUnsupportedReason();  // 每次调用都重新查一次能力
    APP_WARN(
        "Async inference capability query: unsupported (backend: {}, reason: {}); "
        "degrading to the synchronous runInference path",
        getBackendTypeName(), reason.empty() ? std::string("unspecified") : reason);
    return runInference(input_data, std::move(output_data));
}
