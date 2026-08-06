// Immediate-mode UI in the browser: microui -> WebGPU -> canvas.
//
// No GLFW, no C++ runtime. Input comes straight off the html5 event API and
// init is callback-driven, so -sASYNCIFY is not needed.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu.h>

#include "renderer.h"  // pulls in microui.h with C linkage
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

static const float g_clear[3] = { 0.09f, 0.09f, 0.12f };
static float g_speed = 1.0f;

static double g_sim_time;      // seconds
static double g_last_now;      // emscripten_get_now() at the previous frame

// The population. Each cell keeps the constants that make it look like an
// individual -- size, tilt, tint -- plus how it moves; only the position is
// recomputed per frame.
struct Drifter {
    Cell  cell;
    float home_x = 0.0f, home_y = 0.0f;  // world units
    float amp_x  = 0.0f, amp_y  = 0.0f;
    float rate_x = 0.0f, rate_y = 0.0f;
    float phase  = 0.0f;
    float spin   = 0.0f;
};

static std::vector<Drifter> g_field;
static std::vector<Cell>    g_visible;  // rebuilt per frame, kept to avoid churn
static float g_red_count = 120.0f;      // driven by a slider, hence float

// World extent the cells are scattered across, in world units.
constexpr float kWorldX = 900.0f;
constexpr float kWorldY = 620.0f;

// Deterministic, so the same layout comes back after a reset.
static void spawn_field(std::size_t n) {
    static std::mt19937 rng{ 0xB100Du };
    auto uniform = [](float lo, float hi) {
        return std::uniform_real_distribution<float>{ lo, hi };
    };

    while (g_field.size() < n) {
        Drifter d;
        d.home_x = uniform(-kWorldX, kWorldX)(rng);
        d.home_y = uniform(-kWorldY, kWorldY)(rng);
        d.amp_x  = uniform(10.0f, 45.0f)(rng);
        d.amp_y  = uniform(8.0f, 30.0f)(rng);
        d.rate_x = uniform(0.15f, 0.5f)(rng);
        d.rate_y = uniform(0.2f, 0.7f)(rng);
        d.phase  = uniform(0.0f, 6.283f)(rng);
        d.spin   = uniform(-0.25f, 0.25f)(rng);

        // Real red cells are ~7-8um across and vary little; the tilt is what
        // makes a smear look varied, not the size.
        d.cell.kind   = CellKind::RedBlood;
        d.cell.radius = uniform(20.0f, 26.0f)(rng);
        d.cell.angle  = uniform(0.0f, 6.283f)(rng);
        d.cell.squash = uniform(0.55f, 1.0f)(rng);
        d.cell.seed   = uniform(0.0f, 1.0f)(rng);
        g_field.push_back(d);
    }
    if (g_field.size() > n) g_field.resize(n);
}

// Camera. World units are scaled by this on the way to the screen; the UI is
// deliberately unaffected, so the panel stays legible at any zoom.
#define ZOOM_MIN 0.25f
#define ZOOM_MAX 8.0f
static float g_zoom = 1.0f;

static void zoom_by(float factor) {
    g_zoom *= factor;
    if (g_zoom < ZOOM_MIN) g_zoom = ZOOM_MIN;
    if (g_zoom > ZOOM_MAX) g_zoom = ZOOM_MAX;
}

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

    // Over a microui window the wheel still scrolls it; only the empty canvas
    // zooms. hover_root is from the last frame, which is close enough.
    if (g_ctx->hover_root) {
        mu_input_scroll(g_ctx, (int)e->deltaX, (int)e->deltaY);
        return true;
    }

    // Firefox reports lines, and page mode shows up on some configurations.
    double dy = e->deltaY;
    if (e->deltaMode == DOM_DELTA_LINE)      dy *= 16.0;
    else if (e->deltaMode == DOM_DELTA_PAGE) dy *= 100.0;

    zoom_by(expf((float)-dy * 0.0015f));
    return true;
}

// Pinch: the ratio of finger separation between two moves is the zoom factor,
// so it needs no reference to absolute distance or DPI.
static float g_pinch_dist;

static float touch_dist(const EmscriptenTouchEvent *e) {
    const float dx = (float)(e->touches[0].targetX - e->touches[1].targetX);
    const float dy = (float)(e->touches[0].targetY - e->touches[1].targetY);
    return hypotf(dx, dy);
}

static bool on_touch(int type, const EmscriptenTouchEvent *e, void *ud) {
    (void)ud;
    if (type == EMSCRIPTEN_EVENT_TOUCHEND || type == EMSCRIPTEN_EVENT_TOUCHCANCEL ||
        e->numTouches < 2) {
        g_pinch_dist = 0.0f;
        return false;  // let one-finger gestures through
    }

    const float dist = touch_dist(e);
    if (type == EMSCRIPTEN_EVENT_TOUCHMOVE && g_pinch_dist > 0.0f && dist > 0.0f) {
        zoom_by(dist / g_pinch_dist);
    }
    g_pinch_dist = dist;
    return true;  // claim the gesture so the page itself does not zoom
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

// microui ships a grey debug-tool palette. Everything here is still an
// axis-aligned rect -- it has no rounded corners or gradients -- so the look
// comes from colour, weight and spacing: a near-black translucent panel, no
// borders, and one red accent that ties the UI to what is being simulated.
static void apply_style(void) {
    mu_Style *s = g_ctx->style;

    s->size          = mu_vec2(68, 16);  // taller controls, easier to hit
    s->padding       = 8;
    s->spacing       = 7;
    s->title_height  = 30;
    s->scrollbar_size = 10;
    s->thumb_size    = 10;

    s->colors[MU_COLOR_TEXT]        = mu_color(198, 202, 212, 255);
    s->colors[MU_COLOR_BORDER]      = mu_color(0, 0, 0, 0);       // borderless
    s->colors[MU_COLOR_WINDOWBG]    = mu_color(18, 19, 24, 232);  // sits over the sim
    s->colors[MU_COLOR_TITLEBG]     = mu_color(26, 28, 35, 255);
    s->colors[MU_COLOR_TITLETEXT]   = mu_color(236, 238, 243, 255);
    s->colors[MU_COLOR_PANELBG]     = mu_color(0, 0, 0, 0);
    s->colors[MU_COLOR_BUTTON]      = mu_color(38, 41, 51, 255);
    s->colors[MU_COLOR_BUTTONHOVER] = mu_color(50, 54, 68, 255);
    s->colors[MU_COLOR_BUTTONFOCUS] = mu_color(200, 45, 50, 255);  // accent
    s->colors[MU_COLOR_BASE]        = mu_color(30, 32, 40, 255);
    s->colors[MU_COLOR_BASEHOVER]   = mu_color(40, 43, 54, 255);
    s->colors[MU_COLOR_BASEFOCUS]   = mu_color(200, 45, 50, 255);  // accent
    s->colors[MU_COLOR_SCROLLBASE]  = mu_color(24, 26, 32, 255);
    s->colors[MU_COLOR_SCROLLTHUMB] = mu_color(58, 62, 74, 255);
}

static void build_ui(void) {
    if (mu_begin_window(g_ctx, "life", mu_rect(24, 24, 280, 200))) {
        static const int row_label_field[] = { 90, -1 };
        mu_layout_row(g_ctx, 2, row_label_field, 0);

        mu_label(g_ctx, "speed");
        mu_slider(g_ctx, &g_speed, 0.0f, 10.0f);

        mu_label(g_ctx, "red cells");
        mu_slider(g_ctx, &g_red_count, 0.0f, 4000.0f);

        static const int row_full[] = { -1 };
        mu_layout_row(g_ctx, 1, row_full, 0);
        if (mu_button(g_ctx, "reset")) {
            g_speed = 1.0f;
            g_zoom = 1.0f;
            g_red_count = 120.0f;
        }

        mu_end_window(g_ctx);
    }
}

/* ---------------------------------------------------------------- frame -- */

static void configure_surface(int w, int h) {
    WGPUSurfaceConfiguration config = {};
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
    g_sim_time += dt * g_speed;

    mu_begin(g_ctx);
    build_ui();
    mu_end(g_ctx);

    // Cells live in world units around the origin; the camera maps world to
    // screen, zooming about the centre of the canvas.
    spawn_field(static_cast<std::size_t>(g_red_count));

    const float t = static_cast<float>(g_sim_time);
    g_visible.clear();
    g_visible.reserve(g_field.size());
    for (const Drifter &d : g_field) {
        const float wx = d.home_x + std::sin(t * d.rate_x + d.phase) * d.amp_x;
        const float wy = d.home_y + std::sin(t * d.rate_y + d.phase) * d.amp_y;

        Cell c = d.cell;
        c.x      = logical_w * 0.5f + wx * g_zoom;
        c.y      = logical_h * 0.5f + wy * g_zoom;
        c.radius = d.cell.radius * g_zoom;
        c.angle  = d.cell.angle + t * d.spin;
        g_visible.push_back(c);
    }

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

    WGPUSurfaceTexture surface_texture = {};
    wgpuSurfaceGetCurrentTexture(g_surface, &surface_texture);
    if (!surface_texture.texture) return;
    WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, NULL);

    WGPURenderPassColorAttachment color = {};
    color.view       = view;
    color.loadOp     = WGPULoadOp_Clear;
    color.storeOp    = WGPUStoreOp_Store;
    color.clearValue = WGPUColor{ g_clear[0], g_clear[1], g_clear[2], 1.0 };
    // Required since the 2024 webgpu.h revision; zero here is a validation error.
    color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

    WGPURenderPassDescriptor pass_desc = {};
    pass_desc.colorAttachmentCount = 1;
    pass_desc.colorAttachments     = &color;

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_device, NULL);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
    // Cells first: r_end sets scissor rects, and they persist for the rest of
    // the pass. The UI draws on top.
    cells_draw(pass, logical_w, logical_h, g_visible);
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

    g_ctx = new mu_Context();
    mu_init(g_ctx);
    g_ctx->text_width  = text_width;
    g_ctx->text_height = text_height;
    apply_style();

    r_init(g_device, g_queue, kSurfaceFormat);
    cells_init(g_device, g_queue, kSurfaceFormat);

    emscripten_set_mousemove_callback(CANVAS, NULL, 0, on_mouse);
    emscripten_set_mousedown_callback(CANVAS, NULL, 0, on_mouse);
    // Mouse-up on the window so releasing outside the canvas still registers.
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, on_mouse);
    emscripten_set_wheel_callback(CANVAS, NULL, 0, on_wheel);
    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, on_key);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, on_key);
    emscripten_set_touchstart_callback(CANVAS, NULL, 0, on_touch);
    emscripten_set_touchmove_callback(CANVAS, NULL, 0, on_touch);
    emscripten_set_touchend_callback(CANVAS, NULL, 0, on_touch);
    emscripten_set_touchcancel_callback(CANVAS, NULL, 0, on_touch);

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
        printf("requestAdapter failed: %.*s\n", (int)message.length, message.data);
        return;
    }
    g_adapter = adapter;

    WGPUDeviceDescriptor desc = {};
    WGPURequestDeviceCallbackInfo cb = {};
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

    WGPURequestAdapterOptions options = {};
    WGPURequestAdapterCallbackInfo cb = {};
    cb.mode     = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_adapter;
    wgpuInstanceRequestAdapter(g_instance, &options, cb);

    return 0;  // callbacks take it from here
}
