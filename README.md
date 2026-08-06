# life

Immediate-mode UI in the browser: microui on WebGPU, compiled with Emscripten.

All the application code is C. microui is ~1100 lines; the WebGPU backend in
`src/renderer.c` is the other half of the stack. The link step goes through
`em++` because `emdawnwebgpu` implements the `webgpu.h` C API in C++ — the
sources themselves still compile as C99.

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
make deps     # clones microui into third_party/
make serve    # builds, then serves http://localhost:8000/
```

`file://` will not work — the browser blocks the wasm fetch.

## Layout

| Path | Lines | What |
|---|---|---|
| `src/main.c` | ~270 | WebGPU init, html5 input, the UI itself |
| `src/renderer.c` | ~300 | microui command list → batched WebGPU quads |
| `src/shell.html` | ~50 | canvas, HiDPI sizing, WebGPU check |

Built with Emscripten 6.0.6 against the `v20260423` emdawnwebgpu port: 76 KB
wasm, 64 KB JS.

The actual UI is `build_ui()` in `main.c` — about 25 lines. Everything else is
setup you write once.

## How the renderer works

microui doesn't draw anything. It emits a command list per frame — `RECT`,
`TEXT`, `ICON`, `CLIP` — and you rasterise it. `renderer.c` does that in one
pass:

- Every primitive is a quad sampled from a 128×128 R8 atlas (font + 4 icons)
  that ships with microui as `demo/atlas.inl`.
- Quads accumulate into one vertex buffer. The index buffer is built once at
  startup, since the topology never changes.
- `CLIP` commands close the current batch and open a new one; each batch is a
  `setScissorRect` + one `drawIndexed`.

Typical frame: **one pipeline, one bind group, a handful of draw calls.**

## HiDPI

`devicePixelRatio` is rounded to an integer scale. The UI is laid out in
logical units and the shader maps them to the full-resolution framebuffer, with
a **nearest-neighbour** sampler so the bitmap font stays crisp rather than soft.
Scissor rects are converted back to framebuffer pixels in `r_end`.

## Notes on the flags

- `--use-port=emdawnwebgpu` on **both** compile and link.
- No `-sASYNCIFY`. Adapter and device are requested via callbacks; the main loop
  starts once the device exists.
- `emscripten_set_main_loop(frame, 0, 0)` — the `0` means "drive off
  `requestAnimationFrame`", so it tracks 120Hz displays instead of pinning 60.
- `-Os` over `-O2`: this is UI, not a hot inner loop, and size is the point.
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
- Linking with `emcc` fails with *emdawnwebgpu requires C++*. The Makefile
  compiles with `emcc` and links with `em++`; `emcc -sDEFAULT_TO_CXX` also
  works but compiles the `.c` files as C++.
