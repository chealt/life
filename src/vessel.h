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
// Real capillaries run 5-10um across and red cells fold to squeeze through
// them; these cells are rigid discs, so the floor is set above the cell
// diameter instead. Narrower than this and cells would visibly pierce the
// wall.
inline constexpr float kTrunkRadius     = 18.0f;  // um
inline constexpr float kCapillaryRadius = 5.0f;   // um

struct VesselSegment {
    Vec3  a{}, b{};
    float ra = 0.0f, rb = 0.0f;
    // The ring's reference direction at each end. Carried along the branch so
    // consecutive segments agree at the joint they share; deriving it from the
    // axis instead makes the tube twist and crease where the axis turns.
    Vec3  ua{ 1.0f, 0.0f, 0.0f }, ub{ 1.0f, 0.0f, 0.0f };
};

// Total length of every branch added together, in micrometres. All the
// vessels in an adult laid end to end come to roughly 100 000 km, which is
// where the upper limit comes from.
inline constexpr double kBodyVesselLength = 1.0e14;  // um

// A micrometre of capillary holds this many red cells. Capillaries are most
// of the body's vessel length, so this is what the total scales by.
inline constexpr double kCapillaryCellsPerUm =
    3.14159265 * kCapillaryRadius * kCapillaryRadius * kHaematocrit / kRedCellVolume;

struct VesselParams {
    double length = 900.0;  // um, summed over every branch
};

struct VesselStats {
    double requested_length = 0.0;  // um
    double built_length = 0.0;      // um actually turned into geometry
    double true_cells = 0.0;        // red cells the requested length would hold
};

// Builds the vessel tree. Longer vessels fork more, because a branch divides
// once it has run for a set distance and shares what is left with its
// children. Only so much geometry is generated; past that the returned stats
// account for the remainder rather than building it.
VesselStats vessel_build(const VesselParams &params, std::vector<VesselSegment> &out);

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
