// 2D overlay renderer: batched quads sampled from a bitmap atlas.
//
// This draws the UI on top of the 3D scene. It has no idea what a widget is --
// ui.cpp decides that and calls in here with rectangles and strings.
#ifndef RENDERER_H
#define RENDERER_H

#include <cstdint>

#include <webgpu/webgpu.h>

struct UiRect {
    int x = 0, y = 0, w = 0, h = 0;
};

struct UiColor {
    std::uint8_t r = 0, g = 0, b = 0, a = 255;
};

void r_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
            WGPUTextureFormat depth_format);

// Size in logical units; `scale` is the integer factor mapping those to
// framebuffer pixels (scissor rects live in framebuffer space, so we need it).
void r_begin(int width, int height, int scale);
void r_draw_rect(UiRect rect, UiColor color);
void r_draw_text(const char *text, int x, int y, UiColor color);
void r_set_clip_rect(UiRect rect);
void r_end(WGPURenderPassEncoder pass);

int r_text_width(const char *text, int len = -1);
int r_text_height();

#endif
