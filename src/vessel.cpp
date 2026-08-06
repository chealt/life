#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "vessel.h"

namespace {

constexpr float kPi = 3.14159265f;

// A branch divides after running this far, so length and branching are the
// same control: a longer vessel is a more divided one.
constexpr float kForkInterval = 320.0f;  // um
constexpr float kStep         = 40.0f;   // um per segment, sets how smoothly it curves
constexpr int   kMaxDepth     = 7;
constexpr std::size_t kMaxSegments = 30000;

// Murray's law: a parent's cube radius is shared between its children, which
// for a symmetric fork means each child is 2^(-1/3) of the parent.
constexpr float kMurray = 0.7937f;

struct Rng {
    std::uint32_t s;
    std::uint32_t next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
    float unit() { return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f); }
    float range(float lo, float hi) { return lo + unit() * (hi - lo); }
};

void basis_from_axis(const Vec3 &n, Vec3 &u, Vec3 &v) {
    const Vec3 axis = std::abs(n.y) < 0.9f ? Vec3{ 0.0f, 1.0f, 0.0f }
                                           : Vec3{ 1.0f, 0.0f, 0.0f };
    u = normalize(cross(axis, n));
    v = cross(n, u);
}

Vec3 rotate_toward(const Vec3 &dir, const Vec3 &side, float angle) {
    return normalize(dir * std::cos(angle) + side * std::sin(angle));
}

struct Branch {
    Vec3   pos;
    Vec3   dir;
    Vec3   frame;  // reference direction perpendicular to dir
    float  radius;
    double budget;  // um still to run, double because it spans body scale
    int    depth;
    std::uint32_t seed;
};

/* ---------------------------------------------------------- rendering -- */

constexpr int kRings = 24;  // sides around the tube

struct Instance {
    float ax, ay, az;
    float ra;
    float bx, by, bz;
    float rb;
    float uax, uay, uaz;
    float _pad0;
    float ubx, uby, ubz;
    float _pad1;
};
static_assert(sizeof(Instance) == 64, "instance stride is baked into the layout");

WGPUDevice         g_device;
WGPUQueue          g_queue;
WGPURenderPipeline g_pipeline;
WGPUBindGroup      g_bind_group;
WGPUBuffer         g_ring_buffer;
WGPUBuffer         g_index_buffer;
WGPUBuffer         g_instance_buffer;
WGPUBuffer         g_uniform_buffer;
std::uint32_t      g_index_count;

std::vector<Instance> g_instances;

// The wall is drawn as a translucent tube: nearly clear face on, opaque at the
// grazing angles around the silhouette, which is how a glassy tube reads.
const char *kShader =
"struct Uniforms {\n"
"  view_proj: mat4x4f,\n"
"  eye: vec3f,\n"
"  _p0: f32,\n"
"};\n"
"@group(0) @binding(0) var<uniform> u: Uniforms;\n"
"\n"
"struct VSOut {\n"
"  @builtin(position) pos: vec4f,\n"
"  @location(0) world: vec3f,\n"
"  @location(1) normal: vec3f,\n"
"};\n"
"\n"
"@vertex\n"
"fn vs(@location(0) ring: vec3f,\n"
"      @location(1) a: vec3f,\n"
"      @location(2) ra: f32,\n"
"      @location(3) b: vec3f,\n"
"      @location(4) rb: f32,\n"
"      @location(5) ua: vec3f,\n"
"      @location(6) ub: vec3f) -> VSOut {\n"
"  let axis = normalize(b - a);\n"
"  let t = ring.z;\n"
"  // The frame comes from the branch, so the ring lines up with the one in\n"
"  // the neighbouring segment and the tube stays seamless.\n"
"  let uf = normalize(mix(ua, ub, t));\n"
"  let bu = normalize(uf - axis * dot(uf, axis));\n"
"  let bv = cross(axis, bu);\n"
"  let radial = bu * ring.x + bv * ring.y;\n"
"  let centre = mix(a, b, t);\n"
"  let world = centre + radial * mix(ra, rb, t);\n"
"\n"
"  var out: VSOut;\n"
"  out.pos = u.view_proj * vec4f(world, 1.0);\n"
"  out.world = world;\n"
"  out.normal = radial;\n"
"  return out;\n"
"}\n"
"\n"
"@fragment\n"
"fn fs(in: VSOut) -> @location(0) vec4f {\n"
"  let n = normalize(in.normal);\n"
"  let view = normalize(u.eye - in.world);\n"
"  // Fresnel-ish: transparent where we look straight through the wall,\n"
"  // dense where the wall is edge on.\n"
"  let facing = abs(dot(n, view));\n"
"  let rim = pow(1.0 - facing, 2.5);\n"
"\n"
"  let wall = vec3f(0.72, 0.30, 0.34);\n"
"  let lit  = wall * (0.45 + 0.55 * max(dot(n, normalize(vec3f(-0.45, 0.72, 0.52))), 0.0));\n"
"  return vec4f(lit, 0.07 + 0.55 * rim);\n"
"}\n";

WGPUStringView str(const char *s) {
    WGPUStringView v;
    v.data = s;
    v.length = WGPU_STRLEN;
    return v;
}

void create_pipeline(WGPUTextureFormat format, WGPUTextureFormat depth_format) {
    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = str(kShader);
    WGPUShaderModuleDescriptor module_desc = {};
    module_desc.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_device, &module_desc);

    WGPUVertexAttribute ring_attr = {};
    ring_attr.format = WGPUVertexFormat_Float32x3;
    ring_attr.offset = 0;
    ring_attr.shaderLocation = 0;

    std::array<WGPUVertexAttribute, 6> inst{};
    inst[0].format = WGPUVertexFormat_Float32x3;
    inst[0].offset = offsetof(Instance, ax);
    inst[0].shaderLocation = 1;
    inst[1].format = WGPUVertexFormat_Float32;
    inst[1].offset = offsetof(Instance, ra);
    inst[1].shaderLocation = 2;
    inst[2].format = WGPUVertexFormat_Float32x3;
    inst[2].offset = offsetof(Instance, bx);
    inst[2].shaderLocation = 3;
    inst[3].format = WGPUVertexFormat_Float32;
    inst[3].offset = offsetof(Instance, rb);
    inst[3].shaderLocation = 4;
    inst[4].format = WGPUVertexFormat_Float32x3;
    inst[4].offset = offsetof(Instance, uax);
    inst[4].shaderLocation = 5;
    inst[5].format = WGPUVertexFormat_Float32x3;
    inst[5].offset = offsetof(Instance, ubx);
    inst[5].shaderLocation = 6;

    std::array<WGPUVertexBufferLayout, 2> layouts{};
    layouts[0].arrayStride    = sizeof(float) * 3;
    layouts[0].stepMode       = WGPUVertexStepMode_Vertex;
    layouts[0].attributeCount = 1;
    layouts[0].attributes     = &ring_attr;
    layouts[1].arrayStride    = sizeof(Instance);
    layouts[1].stepMode       = WGPUVertexStepMode_Instance;
    layouts[1].attributeCount = inst.size();
    layouts[1].attributes     = inst.data();

    WGPUBlendState blend = {};
    blend.color.operation = WGPUBlendOperation_Add;
    blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend.alpha.operation = WGPUBlendOperation_Add;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;

    WGPUColorTargetState target = {};
    target.format    = format;
    target.blend     = &blend;
    target.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragment = {};
    fragment.module      = module;
    fragment.entryPoint  = str("fs");
    fragment.targetCount = 1;
    fragment.targets     = &target;

    // Tested against the cells so blood in front of the wall stays in front,
    // but not written: the wall is see-through and must not occlude itself.
    WGPUDepthStencilState depth = {};
    depth.format               = depth_format;
    depth.depthWriteEnabled    = WGPUOptionalBool_False;
    depth.depthCompare         = WGPUCompareFunction_Less;
    depth.stencilFront.compare = WGPUCompareFunction_Always;
    depth.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor desc = {};
    desc.layout             = nullptr;
    desc.vertex.module      = module;
    desc.vertex.entryPoint  = str("vs");
    desc.vertex.bufferCount = layouts.size();
    desc.vertex.buffers     = layouts.data();
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode = WGPUCullMode_None;  // both walls are visible
    desc.multisample.count  = 1;
    desc.multisample.mask   = 0xFFFFFFFFu;
    desc.depthStencil       = &depth;
    desc.fragment           = &fragment;

    g_pipeline = wgpuDeviceCreateRenderPipeline(g_device, &desc);
    wgpuShaderModuleRelease(module);
}

}  // namespace

VesselStats vessel_build(const VesselParams &params, std::vector<VesselSegment> &out) {
    out.clear();

    VesselStats stats;
    stats.requested_length = std::max(params.length, static_cast<double>(kStep));

    std::vector<Branch> stack;
    stack.push_back(Branch{ Vec3{ 0.0f, 0.0f, 0.0f }, Vec3{ 0.0f, 0.0f, 1.0f },
                            Vec3{ 1.0f, 0.0f, 0.0f },
                            kTrunkRadius, stats.requested_length, 0, 0x5EEDu });

    // Breadth first. Depth first would spend the whole segment budget running
    // one branch to its end, leaving the rest of the tree unbuilt -- so a
    // longer vessel would show *less* of itself, not more.
    std::size_t head = 0;
    while (head < stack.size() && out.size() < kMaxSegments) {
        Branch b = stack[head++];

        Rng rng{ b.seed | 1u };
        float run = 0.0f;  // distance this branch has covered since it started

        while (b.budget > 0.0 && out.size() < kMaxSegments) {
            const float step = static_cast<float>(std::min<double>(kStep, b.budget));

            // A gentle wander, so vessels do not look like drawn diagrams,
            // plus a push away from the trunk axis. Branches that keep
            // heading outward run into each other far less than ones left to
            // wander freely, though nothing here strictly prevents it.
            Vec3 su, sv;
            basis_from_axis(b.dir, su, sv);
            const Vec3 frame_in = b.frame;

            Vec3 outward{ b.pos.x, b.pos.y, 0.0f };
            const float spread = std::sqrt(dot(outward, outward));
            if (spread > 1e-3f && b.depth > 0) {
                outward = outward * (1.0f / spread);
                b.dir = normalize(b.dir + outward * 0.06f);
            }
            b.dir = normalize(b.dir + su * rng.range(-0.08f, 0.08f) +
                                      sv * rng.range(-0.08f, 0.08f));

            // Parallel transport: keep the reference direction as close to
            // where it was as the axis allows, rather than rebuilding it.
            b.frame = normalize(b.frame - b.dir * dot(b.frame, b.dir));

            const Vec3 next = b.pos + b.dir * step;
            out.push_back(VesselSegment{ b.pos, next, b.radius, b.radius,
                                         frame_in, b.frame });
            b.pos = next;
            b.budget -= step;
            run += step;
            stats.built_length += step;

            if (run < kForkInterval) continue;

            const bool can_fork = b.depth < kMaxDepth &&
                                  b.radius * kMurray > kCapillaryRadius;
            if (!can_fork || b.budget <= kStep) {
                // A branch too narrow to divide still has to yield here.
                // Letting it run its whole budget would hand the entire
                // segment allowance to whichever one happened to come first,
                // which is why the tree used to vanish at long lengths.
                if (b.budget > 0.0) {
                    b.seed = rng.next();
                    stack.push_back(b);
                }
                break;
            }

            {
                Vec3 fu, fv;
                basis_from_axis(b.dir, fu, fv);
                // Split the plane of the fork around the axis, so the tree is
                // not flat.
                const float roll  = rng.range(0.0f, 6.2831853f);
                const Vec3  side  = normalize(fu * std::cos(roll) + fv * std::sin(roll));
                const float angle = rng.range(0.55f, 0.85f);
                const float child_r = b.radius * kMurray;

                // The children share what is left, so the total length over
                // the whole tree is exactly what was asked for.
                const double half = b.budget * 0.5;
                const Vec3 d0 = rotate_toward(b.dir, side, angle);
                const Vec3 d1 = rotate_toward(b.dir, side, -angle);
                stack.push_back(Branch{ b.pos, d0,
                                        normalize(b.frame - d0 * dot(b.frame, d0)),
                                        child_r, half, b.depth + 1, rng.next() });
                stack.push_back(Branch{ b.pos, d1,
                                        normalize(b.frame - d1 * dot(b.frame, d1)),
                                        child_r, half, b.depth + 1, rng.next() });
                break;
            }
        }
    }

    // What was built is counted exactly. Past the geometry budget the
    // remainder is almost all capillary, so it is accounted for at capillary
    // density rather than generated -- the alternative is 10^12 segments.
    const double built_cells = static_cast<double>(vessel_red_cell_count(out));
    const double unbuilt = std::max(0.0, stats.requested_length - stats.built_length);
    stats.true_cells = built_cells + unbuilt * kCapillaryCellsPerUm;
    return stats;
}

float vessel_volume(std::span<const VesselSegment> segments) {
    float total = 0.0f;
    for (const VesselSegment &s : segments) {
        const Vec3 d = s.b - s.a;
        const float len = std::sqrt(dot(d, d));
        // Conical frustum, so tapering is accounted for.
        total += kPi * len * (s.ra * s.ra + s.ra * s.rb + s.rb * s.rb) / 3.0f;
    }
    return total;
}

std::size_t vessel_red_cell_count(std::span<const VesselSegment> segments) {
    const float packed = vessel_volume(segments) * kHaematocrit;
    return static_cast<std::size_t>(packed / kRedCellVolume);
}

void vessel_fill(std::span<const VesselSegment> segments, std::size_t cap,
                 std::vector<Cell> &out) {
    out.clear();
    if (segments.empty() || cap == 0) return;

    const std::size_t wanted = vessel_red_cell_count(segments);
    if (wanted == 0) return;

    // When the true count exceeds what is worth drawing, thin it evenly rather
    // than filling the first segments and leaving the rest empty.
    const float keep = std::min(1.0f, static_cast<float>(cap) /
                                      static_cast<float>(wanted));

    const float volume = vessel_volume(segments);
    if (volume <= 0.0f) return;

    std::uint32_t seed = 0x9E37u;
    for (const VesselSegment &s : segments) {
        const Vec3 d = s.b - s.a;
        const float len = std::sqrt(dot(d, d));
        if (len <= 0.0f) continue;

        const Vec3 axis = d * (1.0f / len);
        Vec3 u, v;
        basis_from_axis(axis, u, v);

        const float seg_volume =
            kPi * len * (s.ra * s.ra + s.ra * s.rb + s.rb * s.rb) / 3.0f;
        const float share = seg_volume * kHaematocrit / kRedCellVolume * keep;

        Rng rng{ (seed = seed * 1664525u + 1013904223u) | 1u };

        // Fractional part carried as a probability, so short segments still
        // hold their share on average instead of rounding to nothing.
        std::size_t n = static_cast<std::size_t>(share);
        if (rng.unit() < share - static_cast<float>(n)) n++;

        for (std::size_t i = 0; i < n; i++) {
            const float t = rng.unit();
            const float radius_here = s.ra + (s.rb - s.ra) * t;
            // sqrt keeps the scatter uniform per unit area rather than
            // crowding the axis.
            const float rr = std::sqrt(rng.unit()) *
                             std::max(0.0f, radius_here - kRedCellRadius);
            const float phi = rng.range(0.0f, 6.2831853f);

            Cell c;
            c.pos = s.a + axis * (len * t) +
                    u * (rr * std::cos(phi)) + v * (rr * std::sin(phi));

            const float uu    = rng.range(-1.0f, 1.0f);
            const float theta = rng.range(0.0f, 6.2831853f);
            const float sr    = std::sqrt(std::max(0.0f, 1.0f - uu * uu));
            c.normal = Vec3{ sr * std::cos(theta), sr * std::sin(theta), uu };

            // Never wider than the lumen: a rigid disc bigger than its vessel
            // reads as a mistake even where a real cell would simply fold.
            c.radius = std::min(kRedCellRadius * rng.range(0.92f, 1.08f),
                                radius_here * 0.82f);
            c.spin   = rng.range(0.0f, 6.2831853f);
            c.seed   = rng.unit();
            c.kind   = CellKind::RedBlood;
            out.push_back(c);

            if (out.size() >= cap) return;
        }
    }
}

void vessel_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
                 WGPUTextureFormat depth_format) {
    g_device = device;
    g_queue  = queue;

    create_pipeline(format, depth_format);

    // A unit tube: a ring of points at each end, joined into a strip of quads.
    std::vector<float> ring;
    ring.reserve(kRings * 2 * 3);
    for (int i = 0; i < kRings; i++) {
        const float a = static_cast<float>(i) / kRings * 2.0f * kPi;
        for (int end = 0; end < 2; end++) {
            ring.push_back(std::cos(a));
            ring.push_back(std::sin(a));
            ring.push_back(static_cast<float>(end));
        }
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(kRings * 6);
    for (int i = 0; i < kRings; i++) {
        const std::uint32_t a0 = static_cast<std::uint32_t>(i * 2);
        const std::uint32_t a1 = a0 + 1;
        const std::uint32_t b0 = static_cast<std::uint32_t>(((i + 1) % kRings) * 2);
        const std::uint32_t b1 = b0 + 1;
        indices.insert(indices.end(), { a0, b0, b1, b1, a1, a0 });
    }
    g_index_count = static_cast<std::uint32_t>(indices.size());

    WGPUBufferDescriptor rb = {};
    rb.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    rb.size  = ring.size() * sizeof(float);
    g_ring_buffer = wgpuDeviceCreateBuffer(g_device, &rb);
    wgpuQueueWriteBuffer(g_queue, g_ring_buffer, 0, ring.data(), rb.size);

    WGPUBufferDescriptor ib = {};
    ib.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
    ib.size  = indices.size() * sizeof(std::uint32_t);
    g_index_buffer = wgpuDeviceCreateBuffer(g_device, &ib);
    wgpuQueueWriteBuffer(g_queue, g_index_buffer, 0, indices.data(), ib.size);

    WGPUBufferDescriptor vb = {};
    vb.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    vb.size  = kMaxSegments * sizeof(Instance);
    g_instance_buffer = wgpuDeviceCreateBuffer(g_device, &vb);

    WGPUBufferDescriptor ub = {};
    ub.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ub.size  = 80;
    g_uniform_buffer = wgpuDeviceCreateBuffer(g_device, &ub);

    WGPUBindGroupEntry entry = {};
    entry.binding = 0;
    entry.buffer  = g_uniform_buffer;
    entry.size    = 80;

    WGPUBindGroupDescriptor bgd = {};
    bgd.layout     = wgpuRenderPipelineGetBindGroupLayout(g_pipeline, 0);
    bgd.entryCount = 1;
    bgd.entries    = &entry;
    g_bind_group = wgpuDeviceCreateBindGroup(g_device, &bgd);
}

void vessel_draw(WGPURenderPassEncoder pass, const Mat4 &view_proj,
                 const Vec3 &eye, std::span<const VesselSegment> segments) {
    std::array<float, 20> uniforms{};
    for (std::size_t i = 0; i < 16; i++) uniforms[i] = view_proj[i];
    uniforms[16] = eye.x;
    uniforms[17] = eye.y;
    uniforms[18] = eye.z;
    wgpuQueueWriteBuffer(g_queue, g_uniform_buffer, 0, uniforms.data(),
                         sizeof(uniforms));

    if (segments.empty()) return;

    g_instances.clear();
    for (const VesselSegment &s : segments) {
        g_instances.push_back(Instance{ s.a.x, s.a.y, s.a.z, s.ra,
                                        s.b.x, s.b.y, s.b.z, s.rb,
                                        s.ua.x, s.ua.y, s.ua.z, 0.0f,
                                        s.ub.x, s.ub.y, s.ub.z, 0.0f });
    }
    wgpuQueueWriteBuffer(g_queue, g_instance_buffer, 0, g_instances.data(),
                         g_instances.size() * sizeof(Instance));

    wgpuRenderPassEncoderSetPipeline(pass, g_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, g_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g_ring_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 1, g_instance_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(pass, g_index_buffer,
                                        WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(pass, g_index_count,
                                     static_cast<std::uint32_t>(g_instances.size()),
                                     0, 0, 0);
}
