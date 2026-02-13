import Gemma from '../build/gemma_cpp_js.mjs';
const gemma = await Gemma();

// Environment-aware path resolution
const weightsFilename = '270m-sfp-it.sbs';
let weightsPath;
if (typeof process !== 'undefined' && process.versions && process.versions.node) {
  // We are in Node.js: Convert the file:// URL to a proper OS file path
  const { fileURLToPath } = await import('node:url');
  weightsPath = fileURLToPath(new URL(weightsFilename, import.meta.url));
} else {
  // We are in the Browser: Use the standard URL string
  weightsPath = new URL(weightsFilename, import.meta.url).href;
}

let lastPercent = -1;
function logDownloadProgress(progress) {
  if (progress.total) {
    const percent = Math.floor((progress.loaded / progress.total) * 100);
    if (percent !== lastPercent) {
      console.log(`Downloading model: ${percent}%`);
      lastPercent = percent;
    }
  } else {
    // Log every 10 MB if total length is unknown
    if (Math.floor(progress.loaded / 10485760) !== Math.floor((progress.loaded - progress.chunkLength) / 10485760)) {
      console.log(`Downloading model: ${Math.floor(progress.loaded / 1048576)} MB`);
    }
  }
}

console.log('Downloading weights and initializing pipeline...');
let model;
try {
  model = await gemma.pipeline(weightsPath, { progress: logDownloadProgress });
  const sentence = 'Max 100 word response. Why is the sky blue?';
  console.log('Generating...');
  const result = await model(sentence);
  console.log(result);
} finally {
  console.log('Cleaning up...');
  if (model) model.delete();
  console.log('Cleanup done.');
}
