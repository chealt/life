// Orbit camera: the viewpoint swings around a target it can also be panned
// through, which is the usual way to inspect something suspended in space.
#ifndef CAMERA_H
#define CAMERA_H

#include <algorithm>

#include "math3d.h"

struct Camera {
    Vec3  target{ 0.0f, 0.0f, 0.0f };
    float distance = 700.0f;
    float yaw      = 0.7f;   // radians, around world Y
    float pitch    = 0.35f;  // radians, clamped short of the poles
    float fov_y    = 0.9f;

    static constexpr float kMinDistance = 30.0f;
    static constexpr float kMaxDistance = 40000.0f;
    // Stop just short of vertical: at the pole the up vector is undefined and
    // the view flips.
    static constexpr float kPitchLimit = 1.5533f;  // ~89 degrees

    Vec3 forward() const {
        const float cp = std::cos(pitch);
        return normalize(Vec3{ cp * std::sin(yaw), std::sin(pitch), cp * std::cos(yaw) });
    }

    Vec3 eye() const { return target + forward() * distance; }

    Vec3 right() const { return normalize(cross(Vec3{ 0.0f, 1.0f, 0.0f }, forward())); }

    Mat4 view_proj(float aspect) const {
        const Mat4 view = mat4_look_at(eye(), target, Vec3{ 0.0f, 1.0f, 0.0f });
        // Far plane scales with distance so the field never clips as you pull
        // back; near plane too, to keep depth precision usable.
        const float far_z  = std::max(4000.0f, distance * 8.0f);
        const float near_z = std::max(0.5f, distance * 0.002f);
        return mat4_mul(mat4_perspective(fov_y, aspect, near_z, far_z), view);
    }

    void orbit(float dx, float dy) {
        yaw   += dx * 0.006f;
        pitch = std::clamp(pitch + dy * 0.006f, -kPitchLimit, kPitchLimit);
    }

    // Pan scales with distance, so dragging moves the same amount on screen
    // whether you are close in or far out.
    void pan(float dx, float dy) {
        const Vec3 r = right();
        const Vec3 u = cross(forward(), r);
        const float k = distance * 0.0016f;
        target = target + r * (-dx * k) + u * (-dy * k);
    }

    void dolly(float factor) {
        distance = std::clamp(distance * factor, kMinDistance, kMaxDistance);
    }
};

#endif
