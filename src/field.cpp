#include <cmath>

#include "field.h"

namespace {

constexpr float kChunkSize = 500.0f;

// A cheap integer hash. The mixing matters more than the constants: chunks
// next to each other must not produce visibly related layouts.
constexpr std::uint32_t hash3(std::int32_t x, std::int32_t y, std::int32_t z) {
    std::uint32_t h = 2166136261u;
    for (std::uint32_t v : { static_cast<std::uint32_t>(x),
                             static_cast<std::uint32_t>(y),
                             static_cast<std::uint32_t>(z) }) {
        h ^= v + 0x9E3779B9u + (h << 6) + (h >> 2);
        h *= 16777619u;
        h ^= h >> 15;
    }
    return h;
}

// xorshift, seeded per chunk. Deterministic and good enough for scatter.
struct Rng {
    std::uint32_t s;
    constexpr std::uint32_t next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
    // 0..1
    float unit() { return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f); }
    float range(float lo, float hi) { return lo + unit() * (hi - lo); }
};

std::int32_t floor_div(float v, float size) {
    return static_cast<std::int32_t>(std::floor(v / size));
}

}  // namespace

void field_gather(const Vec3 &centre, float radius, const FieldParams &params,
                  std::vector<Cell> &out) {
    out.clear();
    if (params.per_chunk <= 0) return;

    const std::int32_t cx = floor_div(centre.x, kChunkSize);
    const std::int32_t cy = floor_div(centre.y, kChunkSize);
    const std::int32_t cz = floor_div(centre.z, kChunkSize);
    const std::int32_t reach =
        static_cast<std::int32_t>(std::ceil(radius / kChunkSize));

    const float radius_sq = radius * radius;

    for (std::int32_t iz = cz - reach; iz <= cz + reach; iz++) {
        for (std::int32_t iy = cy - reach; iy <= cy + reach; iy++) {
            for (std::int32_t ix = cx - reach; ix <= cx + reach; ix++) {
                Rng rng{ hash3(ix, iy, iz) | 1u };  // xorshift must not see 0

                const float base_x = static_cast<float>(ix) * kChunkSize;
                const float base_y = static_cast<float>(iy) * kChunkSize;
                const float base_z = static_cast<float>(iz) * kChunkSize;

                for (int i = 0; i < params.per_chunk; i++) {
                    Cell c;
                    c.pos = Vec3{ base_x + rng.range(0.0f, kChunkSize),
                                  base_y + rng.range(0.0f, kChunkSize),
                                  base_z + rng.range(0.0f, kChunkSize) };

                    const Vec3 d = c.pos - centre;
                    if (dot(d, d) > radius_sq) continue;

                    // A uniform direction on the sphere for the disc's face,
                    // so orientations are not biased toward the poles.
                    const float u     = rng.range(-1.0f, 1.0f);
                    const float theta = rng.range(0.0f, 6.2831853f);
                    const float s     = std::sqrt(std::max(0.0f, 1.0f - u * u));
                    c.normal = Vec3{ s * std::cos(theta), s * std::sin(theta), u };

                    c.radius = rng.range(20.0f, 26.0f);
                    c.spin   = rng.range(0.0f, 6.2831853f);
                    c.seed   = rng.unit();
                    c.kind   = CellKind::RedBlood;
                    out.push_back(c);
                }
            }
        }
    }
}
