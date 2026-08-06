# life

A game-like interface for simulating the human body and its processes.

The aim is to model organs and systems — and the interactions between them — as
something continuously running and directly manipulable, rather than a set of
charts that redraw when you change an input.

Everything runs in the browser, compiled to WebAssembly.

> **Status: foundation only.** What exists today is the rendering and UI
> substrate described below. The simulation itself is not written yet.

See [BUILDING.md](BUILDING.md) for requirements, build, and run instructions.

## The stack

Immediate-mode UI via [microui](https://github.com/rxi/microui), drawn with
WebGPU, compiled with Emscripten.

All the application code is C. microui is ~1100 lines; the WebGPU backend in
`src/renderer.c` is the other half of the stack. The link step goes through
`em++` because `emdawnwebgpu` implements the `webgpu.h` C API in C++ — the
sources themselves still compile as C99.

microui's role is the control panel — sliders, toggles, readouts. It is a
widget toolkit whose every primitive is a quad from a bitmap atlas, so the
simulation's visuals will need their own render pass rather than being pushed
through microui's command list.

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
