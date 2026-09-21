#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#ifdef ALPHA_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <optional>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace {
thread_local std::string error;
std::mutex loader_mutex;
struct Model {
    torch::inductor::AOTIModelPackageLoader loader;
    at::Device device;
    at::Tensor boards, scores;
#ifdef ALPHA_CUDA
    std::optional<c10::cuda::CUDAStream> stream;
#endif
    Model(const char* path, size_t width, bool cuda)
        : loader(path), device(cuda ? at::kCUDA : at::kCPU),
          boards(at::empty({static_cast<int64_t>(width), 80},
              at::TensorOptions().dtype(at::kByte).pinned_memory(cuda))),
          scores(at::empty({static_cast<int64_t>(width), 2},
              at::TensorOptions().dtype(at::kInt).pinned_memory(cuda))) {
#ifdef ALPHA_CUDA
        if (cuda) {
            // Finish loading constants before using them on this worker's stream.
            c10::cuda::getCurrentCUDAStream().synchronize();
            stream = c10::cuda::getStreamFromPool(false, 0);
        }
#else
        TORCH_CHECK(!cuda, "runtime was built without CUDA support");
#endif
    }
};
}

extern "C" {
const char* alpha_model_error() { return error.c_str(); }
void* alpha_model_load(const char* path, size_t width, bool cuda) {
    try {
        // Package loading initializes shared native registries. Serialize setup only;
        // inference remains independent across workers and CUDA streams.
        std::lock_guard<std::mutex> lock(loader_mutex);
        return new Model(path, width, cuda);
    }
    catch (const std::exception& e) { error = e.what(); return nullptr; }
}
void alpha_model_free(void* model) { delete static_cast<Model*>(model); }
int alpha_model_eval(void* model, const uint8_t* boards, const int32_t* scores,
                     float* priors, float* values) {
    try {
        c10::InferenceMode guard;
        auto& m = *static_cast<Model*>(model);
        void* stream_handle = nullptr;
#ifdef ALPHA_CUDA
        c10::cuda::OptionalCUDAStreamGuard stream_guard;
        if (m.stream) {
            stream_guard.reset_stream(*m.stream);
            stream_handle = m.stream->stream();
        }
#endif
        const auto b = m.boards.size(0);
        std::memcpy(m.boards.data_ptr(), boards, b * 80);
        std::memcpy(m.scores.data_ptr(), scores, b * 2 * sizeof(int32_t));
        auto result = m.loader.run({m.boards.to(m.device, /*non_blocking=*/true),
                                    m.scores.to(m.device, /*non_blocking=*/true)}, stream_handle);
        TORCH_CHECK(result.size() == 2, "expected policy and value outputs");
        auto p = result[0].to(at::kCPU).contiguous();
        auto v = result[1].to(at::kCPU).contiguous();
        TORCH_CHECK(p.scalar_type() == at::kFloat && p.numel() == 2 * b * 80,
                    "policy output must be float32 [2B, 80]");
        TORCH_CHECK(v.scalar_type() == at::kFloat && v.numel() == 2 * b,
                    "value output must be float32 [2B]");
        std::memcpy(priors, p.data_ptr<float>(), p.numel() * sizeof(float));
        std::memcpy(values, v.data_ptr<float>(), v.numel() * sizeof(float));
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
}
