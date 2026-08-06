// WebGPU backend for microui: turns the command list into batched quads.
#ifndef RENDERER_H
#define RENDERER_H

#include <webgpu/webgpu.h>

// microui stays C and has no linkage guards of its own, so every C++ include
// of it has to name the linkage.
extern "C" {
#include "microui.h"
}

void r_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format);

// Size in logical units; `scale` is the integer factor mapping those to
// framebuffer pixels (scissor rects live in framebuffer space, so we need it).
void r_begin(int width, int height, int scale);
void r_draw_rect(mu_Rect rect, mu_Color color);
void r_draw_text(const char *text, mu_Vec2 pos, mu_Color color);
void r_draw_icon(int id, mu_Rect rect, mu_Color color);
void r_set_clip_rect(mu_Rect rect);
void r_end(WGPURenderPassEncoder pass);

int r_get_text_width(const char *text, int len);
int r_get_text_height(void);

#endif
