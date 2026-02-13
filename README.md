# gemma.cpp.js

A JavaScript/WebAssembly wrapper for the Gemma.cpp project.

This project uses Emscripten to compile `gemma.cpp` to WebAssembly and provides a simple JavaScript API to interact with the model in a browser or node.js.

## Prerequisites

To build and run this project, make sure you have the following installed:

*   The **Emscripten SDK (emsdk)**. Make sure your environment is properly set up (`source emsdk_env.sh`).
*   **CMake** (version 3.22 or higher).
*   **Make** or another generator (like Ninja).
*   **Node.js** (optional, for running the demo in Node.js environment).

## Configuration and Build

Configure the project using `emcmake` and CMake:

```bash
emcmake cmake -DCMAKE_BUILD_TYPE=Release -B build
```

Then compile the project using `make` to generate the `.mjs` file:

```bash
make -j -C build
```

This will create `gemma_cpp_js.mjs` in the `build` directory.

## Running the Demo

The package comes with demo scripts to quickly test it out.

You need to place a gemma weights file (e.g., [270m-sfp-it.sbs](https://www.kaggle.com/models/google/gemma-3/gemmaCpp/3.0-270m-it-sfp)) in the same directory as the script. To run `demo.mjs` using node, you can use:

```bash
node demo/demo.mjs
```

Or run `demo.html` as a local web server (because of CORS being needed for web workers). In your terminal, use a local server like `http-server` from the project root:

```bash
npx http-server .
```

Then visit the URL in your browser and open `/demo/demo.html`.

## Usage Example

```javascript
import Gemma from './build/gemma_cpp_js.mjs';

const gemma = await Gemma();

const weightsPath = '270m-sfp-it.sbs'; // Path to your weights file

const model = await gemma.pipeline(weightsPath, {
  progress: (progress) => console.log(`Downloading: ${Math.floor(progress.loaded / progress.total * 100)}%`)
});

const response = await model("Why is the sky blue?");
console.log(response);

model.delete();
```
