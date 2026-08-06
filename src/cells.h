// Draws cells as instanced quads, shaded procedurally in the fragment shader.
//
// Separate from renderer.c on purpose: that one exists to rasterise microui's
// command list and everything it draws comes from the font atlas. Simulation
// visuals get their own pipeline.
#ifndef CELLS_H
#define CELLS_H

#include <stdint.h>
#include <webgpu/webgpu.h>

void cells_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format);

// Same begin/add/end shape as the microui renderer. Coordinates are logical
// units with the origin top-left, matching the UI.
void cells_begin(int width, int height);
void cells_add(float x, float y, float radius, uint32_t rgba);
void cells_end(WGPURenderPassEncoder pass);

#endif
