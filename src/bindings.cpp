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
  GemmaPipeline(const std::string& tokenizer_path, const std::string& weights_path) {
    gcpp::LoaderArgs loader(tokenizer_path, weights_path);
    // Disable mmap as it can cause issues in WASM environments.
    loader.map = gcpp::Tristate::kFalse;
    gcpp::InferenceArgs inference;
    gcpp::GemmaArgs args(loader, gcpp::ThreadingArgs(), inference);
    // Only use 16 threads max to avoid overwhelming the browser.
    args.threading.max_threads = 16;
    // Thread pinning is not supported in WASM.
    args.threading.pin = gcpp::Tristate::kFalse;
    ctx = std::make_unique<gcpp::ThreadingContext>(args.threading);
    model = std::make_unique<gcpp::Gemma>(args, *ctx);
    env = std::make_unique<gcpp::MatMulEnv>(*ctx);
  }

  std::string generate(const std::string& prompt, std::optional<int> max_tokens) {
    std::vector<int> tokens = gcpp::WrapAndTokenize(
        model->Tokenizer(), model->ChatTemplate(),
        model->Config().wrapping, 0, prompt);

    gcpp::RuntimeConfig runtime_config;
    model->Inference().CopyTo(runtime_config);
    if (max_tokens.has_value()) {
      runtime_config.max_generated_tokens = max_tokens.value();
    }

    std::string result;
    runtime_config.stream_token = [&result, this](int token, float prob) {
      std::string piece;
      if (model->Tokenizer().Decode({token}, &piece)) {
        result += piece;
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
    std::optional<int> max_tokens;
    emscripten::val callback;
    std::string result;
    GenerateContext(GemmaPipeline* p, std::string pr, std::optional<int> mt, emscripten::val cb)
        : pipeline(p), prompt(std::move(pr)), max_tokens(mt), callback(cb) {}
  };

  static void generate_done(void* arg) {
    auto* ctx = static_cast<GenerateContext*>(arg);
    ctx->callback(ctx->result);
    delete ctx;
  }

  void generateAsync(const std::string& prompt, emscripten::val options, emscripten::val callback) {
    std::optional<int> max_tokens;
    if (options.hasOwnProperty("max_tokens")) {
      max_tokens = options["max_tokens"].as<int>();
    }
    auto* ctx = new GenerateContext(this, prompt, max_tokens, callback);
    std::thread([ctx]() {
      ctx->result = ctx->pipeline->generate(ctx->prompt, ctx->max_tokens);
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
