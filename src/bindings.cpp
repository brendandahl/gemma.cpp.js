#include <iostream>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/proxying.h>
#include <thread>

#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/tokenizer.h"
#include "util/threading_context.h"
#include "ops/matmul.h"

using namespace emscripten;

class GemmaPipeline {
public:
  struct GenerateOptions {
    std::optional<int> max_tokens;
    std::optional<float> temperature;
    std::optional<int> top_k;
    std::optional<bool> ignore_eos;
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

  std::string generate(const std::string& prompt, const GenerateOptions& options) {
    std::vector<int> tokens = gcpp::WrapAndTokenize(
        model->Tokenizer(), model->ChatTemplate(),
        model->Config().wrapping, 0, prompt);

    gcpp::RuntimeConfig runtime_config;
    model->Inference().CopyTo(runtime_config);
    if (options.max_tokens.has_value()) {
      runtime_config.max_generated_tokens = options.max_tokens.value();
    }
    if (options.top_k.has_value()) {
      runtime_config.top_k = options.top_k.value();
    }
    if (options.temperature.has_value()) {
      float temp = options.temperature.value();
      if (temp <= 0.0f) {
        runtime_config.top_k = 1;
        runtime_config.temperature = 1.0f;
      } else {
        runtime_config.temperature = temp;
      }
    }

    if (options.ignore_eos.value_or(false)) {
      int eos_id = model->Config().eos_id;
      int secondary_eos_id = model->Config().secondary_eos_id;
      if (runtime_config.top_k == 1) {
        runtime_config.sample_func = [eos_id, secondary_eos_id](
            size_t /*qi*/, size_t /*pos*/, gcpp::Logits logits, size_t /*worker*/) -> gcpp::TokenAndProb {
          if (eos_id >= 0 && static_cast<size_t>(eos_id) < logits.size()) {
            logits[eos_id] = -1e30f;
          }
          if (secondary_eos_id >= 0 && static_cast<size_t>(secondary_eos_id) < logits.size()) {
            logits[secondary_eos_id] = -1e30f;
          }
          int best_token = 0;
          float best_logit = logits[0];
          for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > best_logit) {
              best_logit = logits[i];
              best_token = static_cast<int>(i);
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
    std::string result;
    runtime_config.batch_stream_token = [&result, prompt_size, this](
        size_t /*query_idx*/, size_t pos, int token, float /*prob*/) {
      if (pos >= prompt_size) {
        std::string piece;
        if (model->Tokenizer().Decode({token}, &piece)) {
          result += piece;
        }
      }
      return true;
    };

    gcpp::KVCache kv_cache(model->Config(), model->Inference(), ctx->allocator);
    gcpp::TimingInfo timing_info;

    model->Generate(runtime_config, hwy::Span<const int>(tokens.data(), tokens.size()), 0, kv_cache, *env, timing_info);

    return result;
  }

  struct GenerateContext {
    GemmaPipeline* pipeline;
    std::string prompt;
    GenerateOptions options;
    emscripten::val callback;
    std::string result;
    GenerateContext(GemmaPipeline* p, std::string pr, GenerateOptions opts, emscripten::val cb)
        : pipeline(p), prompt(std::move(pr)), options(opts), callback(cb) {}
  };

  static void generate_done(void* arg) {
    auto* ctx = static_cast<GenerateContext*>(arg);
    ctx->callback(ctx->result);
    delete ctx;
  }

  void generateAsync(const std::string& prompt, emscripten::val options, emscripten::val callback) {
    GenerateOptions opts;
    if (!options.isUndefined() && !options.isNull()) {
      if (options.hasOwnProperty("max_tokens")) {
        opts.max_tokens = options["max_tokens"].as<int>();
      }
      if (options.hasOwnProperty("temperature")) {
        opts.temperature = options["temperature"].as<float>();
      }
      if (options.hasOwnProperty("top_k")) {
        opts.top_k = options["top_k"].as<int>();
      }
      if (options.hasOwnProperty("ignore_eos")) {
        opts.ignore_eos = options["ignore_eos"].as<bool>();
      }
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
