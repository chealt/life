// A branching blood vessel, and the blood inside it.
//
// One world unit is one micrometre, so the dimensions here are the real ones.
#ifndef VESSEL_H
#define VESSEL_H

#include <cstdint>
#include <span>
#include <vector>

#include <webgpu/webgpu.h>

#include "cells.h"
#include "math3d.h"

// Red cells are ~8um across and ~2um thick, and occupy ~45% of blood by
// volume -- the haematocrit. Those two facts fix how many cells a given
// length of vessel holds.
inline constexpr float kRedCellRadius = 4.0f;    // um
inline constexpr float kRedCellVolume = 90.0f;   // um^3, a real MCV
inline constexpr float kHaematocrit   = 0.45f;

// A venule at the trunk, tapering toward capillary calibre as it divides.
inline constexpr float kTrunkRadius     = 18.0f;  // um
inline constexpr float kCapillaryRadius = 3.5f;   // um

struct VesselSegment {
    Vec3  a{}, b{};
    float ra = 0.0f, rb = 0.0f;
};

struct VesselParams {
    float length = 900.0f;  // um, along the trunk before branching is counted
};

// Builds the vessel tree. Longer vessels fork more, because a branch divides
// once it has run for a set distance.
void vessel_build(const VesselParams &params, std::vector<VesselSegment> &out);

// Total lumen volume, in um^3.
float vessel_volume(std::span<const VesselSegment> segments);

// How many red cells that volume of blood actually contains.
std::size_t vessel_red_cell_count(std::span<const VesselSegment> segments);

// Scatters cells through the lumen. `cap` bounds what is generated, since the
// physiological count can exceed what is worth drawing.
void vessel_fill(std::span<const VesselSegment> segments, std::size_t cap,
                 std::vector<Cell> &out);

void vessel_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
                 WGPUTextureFormat depth_format);

void vessel_draw(WGPURenderPassEncoder pass, const Mat4 &view_proj,
                 const Vec3 &eye, std::span<const VesselSegment> segments);

#endif
