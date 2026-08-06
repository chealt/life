// An unbounded population of cells.
//
// Nothing is stored. Space is divided into cubic chunks, and a chunk's cells
// are derived from a hash of its integer coordinates, so the same place always
// produces the same cells however you arrive at it. Only the chunks near the
// camera are ever generated, which is what makes the extent unbounded rather
// than merely large.
#ifndef FIELD_H
#define FIELD_H

#include <cstdint>
#include <vector>

#include "cells.h"
#include "math3d.h"

struct FieldParams {
    int per_chunk = 24;  // cells generated in each chunk
};

// Fills `out` with the cells near `centre`, out to `radius` world units.
void field_gather(const Vec3 &centre, float radius, const FieldParams &params,
                  std::vector<Cell> &out);

#endif
