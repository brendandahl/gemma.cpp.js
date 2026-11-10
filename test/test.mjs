import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import Gemma from '../build/gemma_cpp_js.mjs';

const weightsPath = fileURLToPath(new URL('../demo/270m-sfp-it.sbs', import.meta.url));
const prompt = 'Explain in one sentence why the sky is blue.';
const maxTokens = 16;

const REQUIRED_METRIC_KEYS = [
  'ttft_ms',
  'prompt_tokens',
  'prefill_tokens',
  'prefill_duration_ms',
  'prefill_tokens_per_sec',
  'tokens_generated',
  'decode_duration_ms',
  'decode_tokens_per_sec',
];

function assertValidMetrics(metrics, expectedTokens) {
  assert.equal(typeof metrics, 'object');
  assert.notEqual(metrics, null);

  for (const key of REQUIRED_METRIC_KEYS) {
    assert.equal(
      typeof metrics[key],
      'number',
      `Expected metric "${key}" to be a number, got ${typeof metrics[key]}`
    );
    assert.ok(
      Number.isFinite(metrics[key]) && metrics[key] > 0,
      `Expected metric "${key}" to be finite and > 0, got ${metrics[key]}`
    );
  }

  assert.equal(metrics.tokens_generated, expectedTokens);
  assert.equal(metrics.prefill_tokens, metrics.prompt_tokens - 1);

  const expectedPrefillTps = metrics.prefill_tokens / (metrics.prefill_duration_ms / 1000);
  const expectedDecodeTps = metrics.tokens_generated / (metrics.decode_duration_ms / 1000);
  assert.ok(Math.abs(metrics.prefill_tokens_per_sec - expectedPrefillTps) < 1e-6);
  assert.ok(Math.abs(metrics.decode_tokens_per_sec - expectedDecodeTps) < 1e-6);
}

console.log('Initializing Gemma pipeline...');
const gemma = await Gemma();
const model = await gemma.pipeline(weightsPath);

try {
  console.log('Test 1: Generation returns { text, metrics } with positive timing metrics...');
  const run1 = await model(prompt, {
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    ignore_eos: true,
  });
  assert.equal(typeof run1, 'object');
  assert.notEqual(run1, null);
  assert.equal(typeof run1.text, 'string');
  assert.ok(run1.text.length > 0, 'Expected non-empty generated text');
  assertValidMetrics(run1.metrics, maxTokens);
  console.log('  Metrics:', JSON.stringify(run1.metrics, null, 2));
  console.log('  PASS');

  console.log('Test 2: Deterministic output and edge-case option clamping...');
  const run2 = await model(prompt, {
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 0,
    ignore_eos: true,
  });
  assert.equal(run2.text, run1.text);
  assertValidMetrics(run2.metrics, maxTokens);
  console.log('  PASS');

  console.log('All tests passed.');
} finally {
  model.delete();
}
