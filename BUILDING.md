# Building

## Requirements

- Emscripten **4.0.10 or newer** — the first release vendoring the
  `emdawnwebgpu` port. The older `-sUSE_WEBGPU` is deprecated and its API does
  not match this code.
- A WebGPU browser (current Chrome or Edge), served over `localhost`.

```sh
git clone https://github.com/emscripten-core/emsdk
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh
```

## Build and run

```sh
make serve    # builds, then serves http://localhost:8000/
```

There is nothing to fetch first: the only third-party content is the font
atlas, and that is generated and committed.

`file://` will not work — the browser blocks the wasm fetch.

Other targets: `make` builds without serving and `make clean` removes
`build/`.

## Editor setup

`.vscode/c_cpp_properties.json` configures IntelliSense for the emscripten
target. It hardcodes the emsdk location to `~/Code/emscripten-core/emsdk`; if
yours lives elsewhere, edit the paths there. Note that the C/C++ extension does
not expand `~`, so paths must be absolute or use `${env:HOME}`.

## Regenerating the font

`src/font_atlas.h` is a signed distance field baked from Roboto Regular
(Apache 2.0). It is committed, so this is only needed to change the typeface
or the baked size:

```sh
curl -sSL -o /tmp/stb_truetype.h \
  https://raw.githubusercontent.com/nothings/stb/master/stb_truetype.h
em++ -std=c++17 -O2 -I/tmp -sNODERAWFS=1 -sENVIRONMENT=node \
  tools/make_font.cpp -o /tmp/make_font.js
node /tmp/make_font.js path/to/Roboto-Regular.ttf > src/font_atlas.h
```

Emscripten is used to build the tool only because the Xcode command line
tools on this machine cannot find their own C++ headers; a working native
`c++` would do just as well.

## Notes on the flags

- `--use-port=emdawnwebgpu` on **both** compile and link.
- No `-sASYNCIFY`. Adapter and device are requested via callbacks; the main loop
  starts once the device exists.
- `emscripten_set_main_loop(frame, 0, 0)` — the `0` means "drive off
  `requestAnimationFrame`", so it tracks 120Hz displays instead of pinning 60.
- `-Os` over `-O2`: sized for a UI, where the binary is the point. This is worth
  revisiting once there is simulation code in a hot loop — `-O3` is the right
  setting for that, and `-sALLOW_MEMORY_GROWTH=1` can cost a hitch when the heap
  reallocates mid-frame.
- No GLFW port. Input comes from `emscripten/html5.h` directly.

## If it doesn't compile

The WebGPU C API churned through 2024–2025. The likely culprits, all in
`renderer.c` and `main.c`:

| Current name | Older name |
|---|---|
| `WGPUTexelCopyTextureInfo` | `WGPUImageCopyTexture` |
| `WGPUTexelCopyBufferLayout` | `WGPUTextureDataLayout` |
| `WGPUShaderSourceWGSL` | `WGPUShaderModuleWGSLDescriptor` |
| `WGPUEmscriptenSurfaceSourceCanvasHTMLSelector` | `WGPUSurfaceDescriptorFromCanvasHTMLSelector` |

Callbacks also gained a second `void*` userdata and `WGPUStringView` replaced
`const char*`. Check `webgpu.h` in your active emsdk if a symbol is missing.

Two more that bite on newer toolchains:

- `emscripten/html5.h` callbacks return `bool`, not the old `EM_BOOL`/`int`.
  A mismatch shows up as *incompatible function pointer types* at every
  `emscripten_set_*_callback` call.
- Linking with `emcc` fails with *emdawnwebgpu requires C++*. Everything under
  `src/` is C++ and builds with `em++`; there is no C left in the project.
