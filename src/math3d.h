// Just enough linear algebra for a perspective camera. Column-major matrices,
// right-handed world, and WebGPU's clip space: z in [0, 1], y up on screen.
#ifndef MATH3D_H
#define MATH3D_H

#include <array>
#include <cmath>

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    constexpr Vec3 operator+(const Vec3 &o) const { return { x + o.x, y + o.y, z + o.z }; }
    constexpr Vec3 operator-(const Vec3 &o) const { return { x - o.x, y - o.y, z - o.z }; }
    constexpr Vec3 operator*(float s) const { return { x * s, y * s, z * s }; }
};

constexpr float dot(const Vec3 &a, const Vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}

inline Vec3 normalize(const Vec3 &v) {
    const float len = std::sqrt(dot(v, v));
    return len > 1e-6f ? v * (1.0f / len) : Vec3{ 0.0f, 0.0f, 1.0f };
}

// Column-major, so m[col * 4 + row] -- the layout WGSL expects for mat4x4f.
using Mat4 = std::array<float, 16>;

inline Mat4 mat4_identity() {
    Mat4 m{};
    m[0] = m[5] = m[10] = m[15] = 1.0f;
    return m;
}

inline Mat4 mat4_mul(const Mat4 &a, const Mat4 &b) {
    Mat4 r{};
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) sum += a[k * 4 + row] * b[c * 4 + k];
            r[c * 4 + row] = sum;
        }
    }
    return r;
}

// Standard look-at: builds the world-to-view transform.
inline Mat4 mat4_look_at(const Vec3 &eye, const Vec3 &target, const Vec3 &up) {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

    Mat4 m = mat4_identity();
    m[0] = s.x;  m[4] = s.y;  m[8]  = s.z;  m[12] = -dot(s, eye);
    m[1] = u.x;  m[5] = u.y;  m[9]  = u.z;  m[13] = -dot(u, eye);
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z; m[14] = dot(f, eye);
    return m;
}

// Perspective with z mapped to [0, 1] rather than OpenGL's [-1, 1]; WebGPU,
// Metal and D3D all use the former.
inline Mat4 mat4_perspective(float fov_y, float aspect, float near_z, float far_z) {
    const float t = 1.0f / std::tan(fov_y * 0.5f);
    Mat4 m{};
    m[0]  = t / aspect;
    m[5]  = t;
    m[10] = far_z / (near_z - far_z);
    m[11] = -1.0f;
    m[14] = (near_z * far_z) / (near_z - far_z);
    return m;
}

#endif
