#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/proxying.h>
#include <thread>
#include <vector>

#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/tokenizer.h"
#include "util/threading_context.h"
#include "ops/matmul.h"

using namespace emscripten;

namespace {

template <typename T>
std::optional<T> getOption(const emscripten::val& options, const char* key) {
  if (!options.hasOwnProperty(key)) {
    return std::nullopt;
  }
  const emscripten::val val = options[key];
  if (val.isUndefined() || val.isNull()) {
    return std::nullopt;
  }
  return val.as<T>();
}

}  // namespace

class GemmaPipeline {
public:
  struct GenerateOptions {
    std::optional<int> max_tokens;
    std::optional<float> temperature;
    std::optional<int> top_k;
    std::optional<bool> ignore_eos;
  };

  struct GenerateResult {
    std::string text;
    size_t prompt_tokens = 0;
    gcpp::TimingInfo timing_info;
  };

  GemmaPipeline(const std::string& tokenizer_path, const std::string& weights_path) {
    gcpp::LoaderArgs loader(tokenizer_path, weights_path);
    // Disable mmap as it can cause issues in WASM environments.
    loader.map = gcpp::Tristate::kFalse;
    gcpp::InferenceArgs inference;
    inference.deterministic = true;
    gcpp::GemmaArgs args(loader, gcpp::ThreadingArgs(), inference);
    // Only use 16 threads max to avoid overwhelming the browser.
    args.threading.max_threads = 16;
    // Thread pinning is not supported in WASM.
    args.threading.pin = gcpp::Tristate::kFalse;
    ctx = std::make_unique<gcpp::ThreadingContext>(args.threading);
    model = std::make_unique<gcpp::Gemma>(args, *ctx);
    env = std::make_unique<gcpp::MatMulEnv>(*ctx);
  }

  GenerateResult generate(const std::string& prompt, const GenerateOptions& options) {
    std::vector<int> tokens = gcpp::WrapAndTokenize(
        model->Tokenizer(), model->ChatTemplate(),
        model->Config().wrapping, 0, prompt);

    const bool ignore_eos = options.ignore_eos.value_or(false);
    const int eos_id = model->Config().eos_id;
    const int secondary_eos_id = model->Config().secondary_eos_id;
    const int vocab_size = static_cast<int>(model->Config().vocab_size);
    const int max_top_k = std::max(1, vocab_size - (ignore_eos ? 2 : 0));

    gcpp::RuntimeConfig runtime_config;
    model->Inference().CopyTo(runtime_config);
    if (options.max_tokens.has_value()) {
      runtime_config.max_generated_tokens =
          static_cast<size_t>(std::max(0, options.max_tokens.value()));
    }
    if (options.top_k.has_value()) {
      runtime_config.top_k = static_cast<size_t>(
          std::clamp(options.top_k.value(), 1, max_top_k));
    } else if (runtime_config.top_k > static_cast<size_t>(max_top_k)) {
      runtime_config.top_k = static_cast<size_t>(max_top_k);
    }
    if (options.temperature.has_value()) {
      float temp = options.temperature.value();
      if (!std::isfinite(temp) || temp <= 0.0f) {
        runtime_config.top_k = 1;
        runtime_config.temperature = 1.0f;
      } else {
        runtime_config.temperature = temp;
      }
    }

    if (ignore_eos) {
      if (runtime_config.top_k == 1) {
        runtime_config.sample_func = [eos_id, secondary_eos_id](
            size_t /*qi*/, size_t /*pos*/, gcpp::Logits logits, size_t /*worker*/) -> gcpp::TokenAndProb {
          int best_token = 0;
          float best_logit = -std::numeric_limits<float>::infinity();
          for (size_t i = 0; i < logits.size(); ++i) {
            int tok = static_cast<int>(i);
            if (tok == eos_id || tok == secondary_eos_id) {
              continue;
            }
            if (logits[i] > best_logit) {
              best_logit = logits[i];
              best_token = tok;
            }
          }
          return gcpp::TokenAndProb{best_token, 1.0f};
        };
      } else {
        runtime_config.accept_token = [eos_id, secondary_eos_id](int token, float /*prob*/) {
          return token != eos_id && token != secondary_eos_id;
        };
      }
    }

    const size_t prompt_size = tokens.size();
    std::string text;
    runtime_config.batch_stream_token = [&text, prompt_size, this](
        size_t /*query_idx*/, size_t pos, int token, float /*prob*/) {
      if (pos >= prompt_size) {
        std::string piece;
        if (model->Tokenizer().Decode({token}, &piece)) {
          text += piece;
        }
      }
      return true;
    };

    gcpp::KVCache kv_cache(model->Config(), model->Inference(), ctx->allocator);
    gcpp::TimingInfo timing_info;

    model->Generate(runtime_config, hwy::Span<const int>(tokens.data(), tokens.size()), 0, kv_cache, *env, timing_info);

    return GenerateResult{std::move(text), prompt_size, timing_info};
  }

  struct GenerateContext {
    GemmaPipeline* pipeline;
    std::string prompt;
    GenerateOptions options;
    emscripten::val callback;
    GenerateResult result;
    GenerateContext(GemmaPipeline* p, std::string pr, GenerateOptions opts, emscripten::val cb)
        : pipeline(p), prompt(std::move(pr)), options(opts), callback(cb) {}
  };

  static void generate_done(void* arg) {
    auto* ctx = static_cast<GenerateContext*>(arg);
    const auto& ti = ctx->result.timing_info;
    const double prefill_tps =
        ti.prefill_duration > 0.0
            ? static_cast<double>(ti.prefill_tokens) / ti.prefill_duration
            : 0.0;
    const double decode_tps =
        ti.generate_duration > 0.0
            ? static_cast<double>(ti.tokens_generated) / ti.generate_duration
            : 0.0;

    emscripten::val metrics = emscripten::val::object();
    metrics.set("ttft_ms", ti.time_to_first_token * 1000.0);
    metrics.set("prompt_tokens", static_cast<double>(ctx->result.prompt_tokens));
    metrics.set("prefill_tokens", static_cast<double>(ti.prefill_tokens));
    metrics.set("prefill_duration_ms", ti.prefill_duration * 1000.0);
    metrics.set("prefill_tokens_per_sec", prefill_tps);
    metrics.set("tokens_generated", static_cast<double>(ti.tokens_generated));
    metrics.set("decode_duration_ms", ti.generate_duration * 1000.0);
    metrics.set("decode_tokens_per_sec", decode_tps);

    emscripten::val response = emscripten::val::object();
    response.set("text", ctx->result.text);
    response.set("metrics", metrics);
    ctx->callback(response);
    delete ctx;
  }

  void generateAsync(const std::string& prompt, emscripten::val options, emscripten::val callback) {
    GenerateOptions opts;
    if (!options.isUndefined() && !options.isNull()) {
      opts.max_tokens = getOption<int>(options, "max_tokens");
      opts.temperature = getOption<float>(options, "temperature");
      opts.top_k = getOption<int>(options, "top_k");
      opts.ignore_eos = getOption<bool>(options, "ignore_eos");
    }
    auto* ctx = new GenerateContext(this, prompt, opts, callback);
    std::thread([ctx]() {
      ctx->result = ctx->pipeline->generate(ctx->prompt, ctx->options);
      emscripten_proxy_async(
          emscripten_proxy_get_system_queue(),
          emscripten_main_runtime_thread_id(),
          generate_done,
          ctx
      );
    }).detach();
  }

private:
  std::unique_ptr<gcpp::ThreadingContext> ctx;
  std::unique_ptr<gcpp::Gemma> model;
  std::unique_ptr<gcpp::MatMulEnv> env;
};

struct CreateContext {
  std::string tokenizer_path;
  std::string weights_path;
  emscripten::val callback;
  std::shared_ptr<GemmaPipeline> pipeline;
  CreateContext(std::string tp, std::string wp, emscripten::val cb)
      : tokenizer_path(std::move(tp)), weights_path(std::move(wp)), callback(cb), pipeline(nullptr) {}
};

static void create_done(void* arg) {
  auto* ctx = static_cast<CreateContext*>(arg);
  ctx->callback(ctx->pipeline);
  delete ctx;
}

// Create a GemmaPipeline on another thread to avoid deadlocking the main browser thread.
// GemmaPipeline will spin up a bunch of other threads.
void createGemmaPipelineAsync(const std::string& weights_path, emscripten::val callback) {
  auto* ctx = new CreateContext("", weights_path, callback);
  std::thread([ctx]() {
    ctx->pipeline = std::make_shared<GemmaPipeline>(ctx->tokenizer_path, ctx->weights_path);
    emscripten_proxy_async(
        emscripten_proxy_get_system_queue(),
        emscripten_main_runtime_thread_id(),
        create_done,
        ctx
    );
  }).detach();
}

EMSCRIPTEN_BINDINGS(gemma_cpp_js) {
  class_<GemmaPipeline>("GemmaPipeline")
      .smart_ptr<std::shared_ptr<GemmaPipeline>>("GemmaPipeline")
      .constructor<std::string, std::string>()
      .function("generateAsync", &GemmaPipeline::generateAsync);

  function("createGemmaPipelineAsync", &createGemmaPipelineAsync);
}
