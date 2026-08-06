// Draws cells as instanced quads oriented in 3D, shaded procedurally.
//
// Each cell is a disc standing in space with its own normal, not a billboard:
// turn the camera and a cell seen edge-on collapses to a sliver, because the
// quad genuinely faces that way.
#ifndef CELLS_H
#define CELLS_H

#include <cstdint>
#include <span>

#include <webgpu/webgpu.h>

#include "math3d.h"

enum class CellKind : std::uint32_t {
    RedBlood = 0,
};

struct Cell {
    Vec3     pos{};
    Vec3     normal{ 0.0f, 0.0f, 1.0f };  // the face the disc presents
    float    radius = 24.0f;
    float    spin   = 0.0f;  // rotation within the disc plane
    float    seed   = 0.0f;  // per-cell variation, 0..1
    CellKind kind   = CellKind::RedBlood;
};

void cells_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
                WGPUTextureFormat depth_format);

void cells_draw(WGPURenderPassEncoder pass, const Mat4 &view_proj,
                const Vec3 &eye, std::span<const Cell> cells);

#endif
