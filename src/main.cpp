// A 3D view onto an unbounded field of cells.
//
// No GLFW, no C++ runtime beyond the standard library. Input comes straight
// off the html5 event API and init is callback-driven, so -sASYNCIFY is not
// needed.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu.h>

#include "camera.h"
#include "cells.h"
#include "field.h"
#include "math3d.h"
#include "renderer.h"
#include "ui.h"

#define CANVAS "#canvas"

// Browsers hand out BGRA8Unorm for canvas surfaces.
constexpr WGPUTextureFormat kSurfaceFormat = WGPUTextureFormat_BGRA8Unorm;
constexpr WGPUTextureFormat kDepthFormat   = WGPUTextureFormat_Depth24Plus;

static WGPUInstance g_instance;
static WGPUAdapter  g_adapter;
static WGPUDevice   g_device;
static WGPUQueue    g_queue;
static WGPUSurface  g_surface;

static WGPUTexture     g_depth_texture;
static WGPUTextureView g_depth_view;

static int g_fb_width, g_fb_height;  // framebuffer pixels
static int g_scale = 1;              // integer UI scale (from devicePixelRatio)

static constexpr float kClear[3] = { 0.045f, 0.05f, 0.07f };

static Camera g_camera;
static double g_last_now;  // emscripten_get_now() at the previous frame

// How far from the camera cells are generated. Beyond this the field simply
// is not built, which is what keeps an unbounded world affordable.
static float g_view_range = 1600.0f;

static FieldParams g_field_params;
static std::vector<Cell> g_cells;

/* ---------------------------------------------------------------- input -- */

static UiInput g_ui_input;
static bool    g_ui_pressed_this_frame;

// Drag state. The camera only sees a drag the UI did not claim on press.
static bool  g_dragging;
static bool  g_drag_pans;   // right button, or shift held
static float g_last_x, g_last_y;

static int css_to_logical(double v) {
    return static_cast<int>(v * emscripten_get_device_pixel_ratio() / g_scale + 0.5);
}

static bool on_mouse(int type, const EmscriptenMouseEvent *e, void *ud) {
    (void)ud;
    const int x = css_to_logical(e->targetX);
    const int y = css_to_logical(e->targetY);

    g_ui_input.mouse_x = x;
    g_ui_input.mouse_y = y;

    switch (type) {
        case EMSCRIPTEN_EVENT_MOUSEDOWN:
            g_ui_input.mouse_down = true;
            g_ui_pressed_this_frame = true;
            // The UI gets first refusal; only then does it become a camera drag.
            if (!ui_captures_mouse(x, y)) {
                g_dragging = true;
                g_drag_pans = (e->button == 2) || (e->button == 1) || e->shiftKey;
                g_last_x = static_cast<float>(x);
                g_last_y = static_cast<float>(y);
            }
            break;

        case EMSCRIPTEN_EVENT_MOUSEUP:
            g_ui_input.mouse_down = false;
            g_dragging = false;
            break;

        case EMSCRIPTEN_EVENT_MOUSEMOVE: {
            if (!g_dragging) break;
            const float dx = static_cast<float>(x) - g_last_x;
            const float dy = static_cast<float>(y) - g_last_y;
            g_last_x = static_cast<float>(x);
            g_last_y = static_cast<float>(y);
            if (g_drag_pans) g_camera.pan(dx, dy);
            else             g_camera.orbit(dx, -dy);
            break;
        }
    }
    return true;
}

static bool on_wheel(int type, const EmscriptenWheelEvent *e, void *ud) {
    (void)type; (void)ud;
    if (ui_captures_mouse(g_ui_input.mouse_x, g_ui_input.mouse_y)) return true;

    // Firefox reports lines, and page mode shows up on some configurations.
    double dy = e->deltaY;
    if (e->deltaMode == DOM_DELTA_LINE)      dy *= 16.0;
    else if (e->deltaMode == DOM_DELTA_PAGE) dy *= 100.0;

    g_camera.dolly(std::exp(static_cast<float>(dy) * 0.0015f));
    return true;
}

// Touch: one finger orbits, two fingers pinch to dolly and drag to pan.
static float g_pinch_dist;
static float g_pinch_x, g_pinch_y;
static bool  g_touch_orbiting;

static float touch_dist(const EmscriptenTouchEvent *e) {
    const float dx = static_cast<float>(e->touches[0].targetX - e->touches[1].targetX);
    const float dy = static_cast<float>(e->touches[0].targetY - e->touches[1].targetY);
    return std::hypot(dx, dy);
}

static bool on_touch(int type, const EmscriptenTouchEvent *e, void *ud) {
    (void)ud;
    if (type == EMSCRIPTEN_EVENT_TOUCHEND || type == EMSCRIPTEN_EVENT_TOUCHCANCEL) {
        g_pinch_dist = 0.0f;
        g_touch_orbiting = false;
        g_ui_input.mouse_down = false;
        return false;
    }

    if (e->numTouches == 1) {
        const int x = css_to_logical(e->touches[0].targetX);
        const int y = css_to_logical(e->touches[0].targetY);
        g_ui_input.mouse_x = x;
        g_ui_input.mouse_y = y;

        if (type == EMSCRIPTEN_EVENT_TOUCHSTART) {
            g_ui_input.mouse_down = true;
            g_ui_pressed_this_frame = true;
            g_touch_orbiting = !ui_captures_mouse(x, y);
            g_last_x = static_cast<float>(x);
            g_last_y = static_cast<float>(y);
        } else if (g_touch_orbiting) {
            g_camera.orbit(static_cast<float>(x) - g_last_x,
                           -(static_cast<float>(y) - g_last_y));
            g_last_x = static_cast<float>(x);
            g_last_y = static_cast<float>(y);
        }
        g_pinch_dist = 0.0f;
        return true;
    }

    if (e->numTouches < 2) return false;

    g_touch_orbiting = false;
    g_ui_input.mouse_down = false;

    const float dist = touch_dist(e);
    const float mid_x = 0.5f * static_cast<float>(e->touches[0].targetX +
                                                  e->touches[1].targetX);
    const float mid_y = 0.5f * static_cast<float>(e->touches[0].targetY +
                                                  e->touches[1].targetY);

    if (type == EMSCRIPTEN_EVENT_TOUCHMOVE && g_pinch_dist > 0.0f && dist > 0.0f) {
        // Separation ratio drives the dolly; the midpoint drags the target.
        g_camera.dolly(g_pinch_dist / dist);
        g_camera.pan((mid_x - g_pinch_x) * 0.5f, (mid_y - g_pinch_y) * 0.5f);
    }
    g_pinch_dist = dist;
    g_pinch_x = mid_x;
    g_pinch_y = mid_y;
    return true;  // claim the gesture so the page itself does not zoom
}

/* ------------------------------------------------------------------- ui -- */

static void build_ui(int width, int height) {
    (void)width; (void)height;

    ui_panel_begin("life", 24, 24, 300);

    ui_category("blood");
    ui_slider_int("red blood cells", &g_field_params.per_chunk, 0, 200);

    ui_category("view");
    ui_slider_float("range", &g_view_range, 400.0f, 6000.0f);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "drawn  %zu", g_cells.size());
    ui_label(buf);

    if (ui_button("reset view")) {
        g_camera = Camera{};
        g_field_params = FieldParams{};
        g_view_range = 1600.0f;
    }

    ui_panel_end();
}

/* ---------------------------------------------------------------- frame -- */

static void configure_surface(int w, int h) {
    WGPUSurfaceConfiguration config = {};
    config.device      = g_device;
    config.format      = kSurfaceFormat;
    config.usage       = WGPUTextureUsage_RenderAttachment;
    config.width       = static_cast<uint32_t>(w);
    config.height      = static_cast<uint32_t>(h);
    config.presentMode = WGPUPresentMode_Fifo;
    config.alphaMode   = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(g_surface, &config);
}

static void create_depth_target(int w, int h) {
    if (g_depth_view) wgpuTextureViewRelease(g_depth_view);
    if (g_depth_texture) wgpuTextureRelease(g_depth_texture);

    WGPUTextureDescriptor desc = {};
    desc.dimension     = WGPUTextureDimension_2D;
    desc.size.width    = static_cast<uint32_t>(w);
    desc.size.height   = static_cast<uint32_t>(h);
    desc.size.depthOrArrayLayers = 1;
    desc.format        = kDepthFormat;
    desc.mipLevelCount = 1;
    desc.sampleCount   = 1;
    desc.usage         = WGPUTextureUsage_RenderAttachment;
    g_depth_texture = wgpuDeviceCreateTexture(g_device, &desc);
    g_depth_view = wgpuTextureCreateView(g_depth_texture, nullptr);
}

static void frame() {
    // The shell's JS keeps the backing store at CSS size * devicePixelRatio.
    int w = 0, h = 0;
    emscripten_get_canvas_element_size(CANVAS, &w, &h);
    if (w <= 0 || h <= 0) return;
    if (w != g_fb_width || h != g_fb_height) {
        g_fb_width = w;
        g_fb_height = h;
        configure_surface(w, h);
        create_depth_target(w, h);
    }

    const int logical_w = g_fb_width / g_scale;
    const int logical_h = g_fb_height / g_scale;

    // Advance on wall-clock delta rather than per frame, so anything animated
    // runs at the same rate on a 60Hz and a 120Hz display.
    const double now = emscripten_get_now() / 1000.0;
    double dt = g_last_now > 0.0 ? now - g_last_now : 0.0;
    g_last_now = now;
    if (dt > 0.1) dt = 0.1;  // a backgrounded tab can hand back a huge delta
    (void)dt;

    g_ui_input.mouse_pressed = g_ui_pressed_this_frame;
    g_ui_pressed_this_frame = false;

    // Cells are generated around wherever the camera is looking, so the field
    // has no edges to reach.
    field_gather(g_camera.target, g_view_range, g_field_params, g_cells);

    ui_begin(g_ui_input, logical_w, logical_h);
    r_begin(logical_w, logical_h, g_scale);
    build_ui(logical_w, logical_h);
    ui_end();

    WGPUSurfaceTexture surface_texture = {};
    wgpuSurfaceGetCurrentTexture(g_surface, &surface_texture);
    if (!surface_texture.texture) return;
    WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, nullptr);

    WGPURenderPassColorAttachment color = {};
    color.view       = view;
    color.loadOp     = WGPULoadOp_Clear;
    color.storeOp    = WGPUStoreOp_Store;
    color.clearValue = WGPUColor{ kClear[0], kClear[1], kClear[2], 1.0 };
    // Required since the 2024 webgpu.h revision; zero here is a validation error.
    color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

    WGPURenderPassDepthStencilAttachment depth = {};
    depth.view              = g_depth_view;
    depth.depthLoadOp       = WGPULoadOp_Clear;
    depth.depthStoreOp      = WGPUStoreOp_Store;
    depth.depthClearValue   = 1.0f;

    WGPURenderPassDescriptor pass_desc = {};
    pass_desc.colorAttachmentCount = 1;
    pass_desc.colorAttachments     = &color;
    pass_desc.depthStencilAttachment = &depth;

    const float aspect = static_cast<float>(g_fb_width) /
                         static_cast<float>(g_fb_height);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_device, nullptr);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
    // Cells first: r_end sets scissor rects, and they persist for the rest of
    // the pass. The UI draws on top.
    cells_draw(pass, g_camera.view_proj(aspect), g_camera.eye(), g_cells);
    r_end(pass);
    wgpuRenderPassEncoderEnd(pass);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(g_queue, 1, &commands);

    // No wgpuSurfacePresent() on the web -- the browser composites the canvas.
    wgpuCommandBufferRelease(commands);
    wgpuRenderPassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(view);
}

/* ----------------------------------------------------------------- init -- */

static void start() {
    const double dpr = emscripten_get_device_pixel_ratio();
    g_scale = std::max(1, static_cast<int>(dpr + 0.5));

    r_init(g_device, g_queue, kSurfaceFormat, kDepthFormat);
    cells_init(g_device, g_queue, kSurfaceFormat, kDepthFormat);

    emscripten_set_mousemove_callback(CANVAS, nullptr, 0, on_mouse);
    emscripten_set_mousedown_callback(CANVAS, nullptr, 0, on_mouse);
    // Mouse-up on the window so releasing outside the canvas still registers.
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, 0, on_mouse);
    emscripten_set_wheel_callback(CANVAS, nullptr, 0, on_wheel);
    emscripten_set_touchstart_callback(CANVAS, nullptr, 0, on_touch);
    emscripten_set_touchmove_callback(CANVAS, nullptr, 0, on_touch);
    emscripten_set_touchend_callback(CANVAS, nullptr, 0, on_touch);
    emscripten_set_touchcancel_callback(CANVAS, nullptr, 0, on_touch);

    emscripten_set_main_loop(frame, 0, 0);  // 0 => drive off requestAnimationFrame
}

static void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
                      WGPUStringView message, void *ud1, void *ud2) {
    (void)ud1; (void)ud2;
    if (status != WGPURequestDeviceStatus_Success) {
        std::printf("requestDevice failed: %.*s\n", (int)message.length, message.data);
        return;
    }
    g_device = device;
    g_queue  = wgpuDeviceGetQueue(g_device);

    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas_desc = {};
    canvas_desc.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
    canvas_desc.selector    = WGPUStringView{ CANVAS, WGPU_STRLEN };

    WGPUSurfaceDescriptor surface_desc = {};
    surface_desc.nextInChain = &canvas_desc.chain;
    g_surface = wgpuInstanceCreateSurface(g_instance, &surface_desc);

    start();
}

static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       WGPUStringView message, void *ud1, void *ud2) {
    (void)ud1; (void)ud2;
    if (status != WGPURequestAdapterStatus_Success) {
        std::printf("requestAdapter failed: %.*s\n", (int)message.length, message.data);
        return;
    }
    g_adapter = adapter;

    WGPUDeviceDescriptor desc = {};
    WGPURequestDeviceCallbackInfo cb = {};
    cb.mode     = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_device;
    wgpuAdapterRequestDevice(g_adapter, &desc, cb);
}

int main() {
    g_instance = wgpuCreateInstance(nullptr);
    if (!g_instance) {
        std::printf("wgpuCreateInstance failed -- no WebGPU in this browser?\n");
        return 1;
    }

    WGPURequestAdapterOptions options = {};
    WGPURequestAdapterCallbackInfo cb = {};
    cb.mode     = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_adapter;
    wgpuInstanceRequestAdapter(g_instance, &options, cb);

    return 0;  // callbacks take it from here
}
