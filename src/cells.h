// Draws cells as instanced quads, shaded procedurally in the fragment shader.
//
// Separate from renderer.cpp on purpose: that one exists to rasterise microui's
// command list and everything it draws comes from the font atlas. Simulation
// visuals get their own pipeline.
#ifndef CELLS_H
#define CELLS_H

#include <cstdint>
#include <span>

#include <webgpu/webgpu.h>

enum class CellKind : std::uint32_t {
    RedBlood = 0,
};

struct Cell {
    float    x      = 0.0f;   // centre, world units
    float    y      = 0.0f;
    float    radius = 24.0f;  // world units
    float    angle  = 0.0f;   // radians, rotates the squash axis
    float    squash = 1.0f;   // 1 = face on, towards 0 = seen edge on
    float    seed   = 0.0f;   // per-cell variation, 0..1
    CellKind kind   = CellKind::RedBlood;
};

void cells_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format);

// Coordinates are logical units with the origin top-left, matching the UI.
// Cells are submitted as one span per frame rather than one call per cell.
void cells_draw(WGPURenderPassEncoder pass, int width, int height,
                std::span<const Cell> cells);

#endif
