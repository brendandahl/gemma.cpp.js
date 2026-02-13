
// The keep-alive hacks are needed to keep node.js from exiting while the
// threads are running, but main thread is idle.
// TODO: Remove these hacks when Emscripten provides a way to keep the main thread alive.
function withKeepAlive(execute) {
  const keepAlive = setInterval(() => { }, 100);
  return new Promise((resolve) => {
    execute((result) => {
      clearInterval(keepAlive);
      resolve(result);
    });
  });
}

Module['pipeline'] = async (weightsPath, options = {}) => {
  let data;
  const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

  let existsInMemfs = false;
  try {
    if (typeof FS !== 'undefined' && FS.analyzePath) {
      existsInMemfs = FS.analyzePath(weightsPath).exists;
    }
  } catch (e) {
    // Ignored, path might be invalid for FS
  }

  const memfsPath = existsInMemfs ? weightsPath : '/weights.sbs';

  if (!existsInMemfs) {
    if (isNode) {
      const fs = await import('fs');
      const readFileSync = fs.readFileSync || (fs.default && fs.default.readFileSync);
      if (!readFileSync) {
        throw new Error("Could not find readFileSync on fs module");
      }
      data = readFileSync(weightsPath);
      if (options.progress) {
        options.progress({ loaded: data.length, total: data.length, chunkLength: data.length });
      }
    } else {
      const response = await fetch(weightsPath);
      if (!response.ok) {
        throw new Error(`Failed to fetch weights: ${response.status}`);
      }

      const contentLength = response.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength, 10) : 0;
      const reader = response.body.getReader();

      let loaded = 0;
      const chunks = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
        loaded += value.length;
        if (options.progress) {
          options.progress({ loaded, total, chunkLength: value.length });
        }
      }

      const buffer = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
      }
      data = buffer;
    }

    if (typeof FS !== 'undefined') {
      FS.writeFile(memfsPath, data);
    } else {
      throw new Error("Emscripten FS is not available");
    }
  }

  const pipeline = await withKeepAlive((cb) => {
    Module.createGemmaPipelineAsync(memfsPath, cb);
  });

  const model = (prompt, options = {}) => {
    return withKeepAlive((cb) => {
      pipeline.generateAsync(prompt, options, cb);
    });
  };

  model.delete = () => {
    pipeline.delete();
  };

  return model;
};
