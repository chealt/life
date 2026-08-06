// Immediate-mode UI in the browser: microui -> WebGPU -> canvas.
//
// No GLFW, no C++ runtime. Input comes straight off the html5 event API and
// init is callback-driven, so -sASYNCIFY is not needed.

#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu.h>

#include "microui.h"
#include "renderer.h"
#include "cells.h"

#define CANVAS "#canvas"

// Browsers hand out BGRA8Unorm for canvas surfaces.
static const WGPUTextureFormat kSurfaceFormat = WGPUTextureFormat_BGRA8Unorm;

static WGPUInstance g_instance;
static WGPUAdapter  g_adapter;
static WGPUDevice   g_device;
static WGPUQueue    g_queue;
static WGPUSurface  g_surface;

static mu_Context *g_ctx;
static int g_fb_width, g_fb_height;  // framebuffer pixels
static int g_scale = 1;              // integer UI scale (from devicePixelRatio)

// Demo state, so there is something to drive from the UI.
static float g_clear[3] = { 0.09f, 0.09f, 0.12f };
static float g_speed = 1.0f;
static int   g_running = 1;

// The first thing being simulated: one red blood cell, drifting.
static float  g_cell_radius = 48.0f;
static double g_sim_time;      // seconds, advanced only while running
static double g_last_now;      // emscripten_get_now() at the previous frame

/* ---------------------------------------------------------------- input -- */

static int css_to_logical(double v) {
    return (int)(v * emscripten_get_device_pixel_ratio() / g_scale + 0.5);
}

static bool on_mouse(int type, const EmscriptenMouseEvent *e, void *ud) {
    (void)ud;
    int x = css_to_logical(e->targetX);
    int y = css_to_logical(e->targetY);
    int btn = e->button == 1 ? MU_MOUSE_MIDDLE
            : e->button == 2 ? MU_MOUSE_RIGHT
                             : MU_MOUSE_LEFT;
    switch (type) {
        case EMSCRIPTEN_EVENT_MOUSEMOVE: mu_input_mousemove(g_ctx, x, y); break;
        case EMSCRIPTEN_EVENT_MOUSEDOWN: mu_input_mousedown(g_ctx, x, y, btn); break;
        case EMSCRIPTEN_EVENT_MOUSEUP:   mu_input_mouseup(g_ctx, x, y, btn); break;
    }
    return true;
}

static bool on_wheel(int type, const EmscriptenWheelEvent *e, void *ud) {
    (void)type; (void)ud;
    mu_input_scroll(g_ctx, (int)e->deltaX, (int)e->deltaY);
    return true;
}

static int map_key(const char *code) {
    if (!strcmp(code, "ShiftLeft")   || !strcmp(code, "ShiftRight"))   return MU_KEY_SHIFT;
    if (!strcmp(code, "ControlLeft") || !strcmp(code, "ControlRight")) return MU_KEY_CTRL;
    if (!strcmp(code, "AltLeft")     || !strcmp(code, "AltRight"))     return MU_KEY_ALT;
    if (!strcmp(code, "Backspace")) return MU_KEY_BACKSPACE;
    if (!strcmp(code, "Enter"))     return MU_KEY_RETURN;
    return 0;
}

static bool on_key(int type, const EmscriptenKeyboardEvent *e, void *ud) {
    (void)ud;
    int key = map_key(e->code);
    if (key) {
        if (type == EMSCRIPTEN_EVENT_KEYDOWN) mu_input_keydown(g_ctx, key);
        else                                  mu_input_keyup(g_ctx, key);
        return true;
    }
    // Printable characters arrive here as a one-glyph `key` string.
    if (type == EMSCRIPTEN_EVENT_KEYDOWN && strlen(e->key) == 1) {
        char text[2] = { e->key[0], 0 };
        mu_input_text(g_ctx, text);
        return true;
    }
    return false;
}

/* ------------------------------------------------------------------- ui -- */

static void build_ui(void) {
    if (mu_begin_window(g_ctx, "life", mu_rect(24, 24, 300, 210))) {
        mu_layout_row(g_ctx, 2, (int[]){ 90, -1 }, 0);

        mu_label(g_ctx, "speed");
        mu_slider(g_ctx, &g_speed, 0.0f, 10.0f);
        mu_label(g_ctx, "cell size");
        mu_slider(g_ctx, &g_cell_radius, 8.0f, 160.0f);

        mu_label(g_ctx, "red");
        mu_slider(g_ctx, &g_clear[0], 0.0f, 1.0f);
        mu_label(g_ctx, "green");
        mu_slider(g_ctx, &g_clear[1], 0.0f, 1.0f);
        mu_label(g_ctx, "blue");
        mu_slider(g_ctx, &g_clear[2], 0.0f, 1.0f);

        mu_layout_row(g_ctx, 1, (int[]){ -1 }, 0);
        mu_checkbox(g_ctx, "running", &g_running);
        if (mu_button(g_ctx, "reset")) {
            g_clear[0] = 0.09f; g_clear[1] = 0.09f; g_clear[2] = 0.12f;
            g_speed = 1.0f;
            g_cell_radius = 48.0f;
        }

        mu_end_window(g_ctx);
    }
}

/* ---------------------------------------------------------------- frame -- */

static void configure_surface(int w, int h) {
    WGPUSurfaceConfiguration config = {0};
    config.device      = g_device;
    config.format      = kSurfaceFormat;
    config.usage       = WGPUTextureUsage_RenderAttachment;
    config.width       = (uint32_t)w;
    config.height      = (uint32_t)h;
    config.presentMode = WGPUPresentMode_Fifo;
    config.alphaMode   = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(g_surface, &config);
}

static void frame(void) {
    // The shell's JS keeps the backing store at CSS size * devicePixelRatio.
    int w = 0, h = 0;
    emscripten_get_canvas_element_size(CANVAS, &w, &h);
    if (w <= 0 || h <= 0) return;
    if (w != g_fb_width || h != g_fb_height) {
        g_fb_width = w;
        g_fb_height = h;
        configure_surface(w, h);
    }

    const int logical_w = g_fb_width / g_scale;
    const int logical_h = g_fb_height / g_scale;

    // Advance on wall-clock delta rather than per frame, so the drift runs at
    // the same rate on a 60Hz and a 120Hz display.
    const double now = emscripten_get_now() / 1000.0;
    double dt = g_last_now > 0.0 ? now - g_last_now : 0.0;
    g_last_now = now;
    if (dt > 0.1) dt = 0.1;  // a backgrounded tab can hand back a huge delta
    if (g_running) g_sim_time += dt * g_speed;

    mu_begin(g_ctx);
    build_ui();
    mu_end(g_ctx);

    // One cell for now, drifting around the middle of the canvas.
    const float t  = (float)g_sim_time;
    const float cx = logical_w * 0.5f + sinf(t * 0.6f) * (logical_w * 0.12f);
    const float cy = logical_h * 0.5f + sinf(t * 0.9f) * (logical_h * 0.08f);
    cells_begin(logical_w, logical_h);
    cells_add(cx, cy, g_cell_radius, 0xFF322DC8u);  // RGBA8: deep red, opaque

    r_begin(logical_w, logical_h, g_scale);
    mu_Command *cmd = NULL;
    while (mu_next_command(g_ctx, &cmd)) {
        switch (cmd->type) {
            case MU_COMMAND_RECT: r_draw_rect(cmd->rect.rect, cmd->rect.color); break;
            case MU_COMMAND_TEXT: r_draw_text(cmd->text.str, cmd->text.pos, cmd->text.color); break;
            case MU_COMMAND_ICON: r_draw_icon(cmd->icon.id, cmd->icon.rect, cmd->icon.color); break;
            case MU_COMMAND_CLIP: r_set_clip_rect(cmd->clip.rect); break;
        }
    }

    WGPUSurfaceTexture surface_texture = {0};
    wgpuSurfaceGetCurrentTexture(g_surface, &surface_texture);
    if (!surface_texture.texture) return;
    WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, NULL);

    WGPURenderPassColorAttachment color = {0};
    color.view       = view;
    color.loadOp     = WGPULoadOp_Clear;
    color.storeOp    = WGPUStoreOp_Store;
    color.clearValue = (WGPUColor){ g_clear[0], g_clear[1], g_clear[2], 1.0 };
    // Required since the 2024 webgpu.h revision; zero here is a validation error.
    color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

    WGPURenderPassDescriptor pass_desc = {0};
    pass_desc.colorAttachmentCount = 1;
    pass_desc.colorAttachments     = &color;

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_device, NULL);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
    // Cells first: r_end sets scissor rects, and they persist for the rest of
    // the pass. The UI draws on top.
    cells_end(pass);
    r_end(pass);
    wgpuRenderPassEncoderEnd(pass);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(g_queue, 1, &commands);

    // No wgpuSurfacePresent() on the web -- the browser composites the canvas.
    wgpuCommandBufferRelease(commands);
    wgpuRenderPassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(view);
}

/* ----------------------------------------------------------------- init -- */

static int text_width(mu_Font font, const char *text, int len) {
    (void)font;
    if (len == -1) len = (int)strlen(text);
    return r_get_text_width(text, len);
}

static int text_height(mu_Font font) {
    (void)font;
    return r_get_text_height();
}

static void start(void) {
    double dpr = emscripten_get_device_pixel_ratio();
    g_scale = (int)(dpr + 0.5);
    if (g_scale < 1) g_scale = 1;

    g_ctx = malloc(sizeof(mu_Context));
    mu_init(g_ctx);
    g_ctx->text_width  = text_width;
    g_ctx->text_height = text_height;

    r_init(g_device, g_queue, kSurfaceFormat);
    cells_init(g_device, g_queue, kSurfaceFormat);

    emscripten_set_mousemove_callback(CANVAS, NULL, 0, on_mouse);
    emscripten_set_mousedown_callback(CANVAS, NULL, 0, on_mouse);
    // Mouse-up on the window so releasing outside the canvas still registers.
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, on_mouse);
    emscripten_set_wheel_callback(CANVAS, NULL, 0, on_wheel);
    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, on_key);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, on_key);

    emscripten_set_main_loop(frame, 0, 0);  // 0 => drive off requestAnimationFrame
}

static void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
                      WGPUStringView message, void *ud1, void *ud2) {
    (void)ud1; (void)ud2;
    if (status != WGPURequestDeviceStatus_Success) {
        printf("requestDevice failed: %.*s\n", (int)message.length, message.data);
        return;
    }
    g_device = device;
    g_queue  = wgpuDeviceGetQueue(g_device);

    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas_desc = {0};
    canvas_desc.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
    canvas_desc.selector    = (WGPUStringView){ CANVAS, WGPU_STRLEN };

    WGPUSurfaceDescriptor surface_desc = {0};
    surface_desc.nextInChain = &canvas_desc.chain;
    g_surface = wgpuInstanceCreateSurface(g_instance, &surface_desc);

    start();
}

static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       WGPUStringView message, void *ud1, void *ud2) {
    (void)ud1; (void)ud2;
    if (status != WGPURequestAdapterStatus_Success) {
        printf("requestAdapter failed: %.*s\n", (int)message.length, message.data);
        return;
    }
    g_adapter = adapter;

    WGPUDeviceDescriptor desc = {0};
    WGPURequestDeviceCallbackInfo cb = {0};
    cb.mode     = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_device;
    wgpuAdapterRequestDevice(g_adapter, &desc, cb);
}

int main(void) {
    g_instance = wgpuCreateInstance(NULL);
    if (!g_instance) {
        printf("wgpuCreateInstance failed -- no WebGPU in this browser?\n");
        return 1;
    }

    WGPURequestAdapterOptions options = {0};
    WGPURequestAdapterCallbackInfo cb = {0};
    cb.mode     = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_adapter;
    wgpuInstanceRequestAdapter(g_instance, &options, cb);

    return 0;  // callbacks take it from here
}
