#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cells.h"

namespace {

constexpr std::size_t kMaxCells = 200000;

// What actually reaches the GPU: the disc's centre and its two in-plane basis
// vectors, so the vertex shader can place the quad without any trigonometry.
struct Instance {
    float         px, py, pz;
    float         radius;
    float         ux, uy, uz;
    float         seed;
    float         vx, vy, vz;
    std::uint32_t kind;
};
static_assert(sizeof(Instance) == 48, "instance stride is baked into the layout");

WGPUDevice         g_device;
WGPUQueue          g_queue;
WGPURenderPipeline g_pipeline;
WGPUBindGroup      g_bind_group;
WGPUBuffer         g_corner_buffer;
WGPUBuffer         g_index_buffer;
WGPUBuffer         g_instance_buffer;
WGPUBuffer         g_uniform_buffer;

std::vector<Instance> g_instances;

// A red blood cell is a biconcave disc: a torus of haemoglobin around a thin
// central dimple. Rather than model the geometry, the shader reconstructs the
// thickness of the cell at each pixel and treats colour as light passing
// through that much haemoglobin -- thick reads deep red, thin reads pale. That
// is why a real one has a pale centre, and it falls out for free here.
const char *kShader =
"struct Uniforms {\n"
"  view_proj: mat4x4f,\n"
"  eye: vec3f,\n"
"  _p0: f32,\n"
"  cam_right: vec3f,\n"
"  _p1: f32,\n"
"  cam_up: vec3f,\n"
"  _p2: f32,\n"
"};\n"
"@group(0) @binding(0) var<uniform> u: Uniforms;\n"
"\n"
"// Half-thickness of a red cell relative to its radius: roughly 2.5um across\n"
"// a 8um disc.\n"
"const K: f32 = 0.30;\n"
"\n"
"struct VSOut {\n"
"  @builtin(position) pos: vec4f,\n"
"  @location(0) world: vec3f,\n"
"  @location(1) @interpolate(flat) centre: vec3f,\n"
"  @location(2) @interpolate(flat) radius: f32,\n"
"  @location(3) @interpolate(flat) bu: vec3f,\n"
"  @location(4) @interpolate(flat) bv: vec3f,\n"
"  @location(5) @interpolate(flat) bn: vec3f,\n"
"  @location(6) @interpolate(flat) kind: u32,\n"
"  @location(7) @interpolate(flat) seed: f32,\n"
"};\n"
"\n"
"struct FSOut {\n"
"  @location(0) color: vec4f,\n"
"  @builtin(frag_depth) depth: f32,\n"
"};\n"
"\n"
"// The quad is a camera-facing billboard just large enough to contain the\n"
"// cell; the solid itself is found per pixel in the fragment shader, so the\n"
"// silhouette is correct from any angle rather than being the quad.\n"
"@vertex\n"
"fn vs(@location(0) corner: vec2f,\n"
"      @location(1) centre: vec3f,\n"
"      @location(2) radius: f32,\n"
"      @location(3) basis_u: vec3f,\n"
"      @location(4) seed: f32,\n"
"      @location(5) basis_v: vec3f,\n"
"      @location(6) kind: u32) -> VSOut {\n"
"  let world = centre + (u.cam_right * corner.x + u.cam_up * corner.y) * radius * 1.05;\n"
"  var out: VSOut;\n"
"  out.pos = u.view_proj * vec4f(world, 1.0);\n"
"  out.world = world;\n"
"  out.centre = centre;\n"
"  out.radius = radius;\n"
"  out.bu = basis_u;\n"
"  out.bv = basis_v;\n"
"  out.bn = normalize(cross(basis_u, basis_v));\n"
"  out.kind = kind;\n"
"  out.seed = seed;\n"
"  return out;\n"
"}\n"
"\n"
"// Slope of the upper surface, in units of the radius: the sqrt term rounds\n"
"// the disc off and the gaussian presses the dimple in. Only the derivative\n"
"// is needed, since the surface normal is all this is used for.\n"
"fn profile_slope(rho: f32) -> f32 {\n"
"  let s = sqrt(max(1.0 - rho * rho, 1e-4));\n"
"  return K * (-rho / s + 6.82 * rho * exp(-5.5 * rho * rho));\n"
"}\n"
"\n"
"@fragment\n"
"fn fs(in: VSOut) -> FSOut {\n"
"  // Into the cell's own frame, with the radius scaled out.\n"
"  let rel = (in.world - in.centre) / in.radius;\n"
"  let ro = vec3f(dot(rel, in.bu), dot(rel, in.bv), dot(rel, in.bn));\n"
"  let eye_rel = (u.eye - in.centre) / in.radius;\n"
"  let eo = vec3f(dot(eye_rel, in.bu), dot(eye_rel, in.bv), dot(eye_rel, in.bn));\n"
"  let rd = normalize(ro - eo);\n"
"\n"
"  // Intersect the bounding oblate spheroid analytically: squash z and it is\n"
"  // an ordinary ray-sphere test.\n"
"  let os = vec3f(eo.x, eo.y, eo.z / K);\n"
"  let ds = vec3f(rd.x, rd.y, rd.z / K);\n"
"  let a = dot(ds, ds);\n"
"  let b = 2.0 * dot(os, ds);\n"
"  let c = dot(os, os) - 1.0;\n"
"  let disc = b * b - 4.0 * a * c;\n"
"  if (disc <= 0.0) { discard; }\n"
"\n"
"  let sq = sqrt(disc);\n"
"  let t0 = (-b - sq) / (2.0 * a);\n"
"  let t1 = (-b + sq) / (2.0 * a);\n"
"  if (t1 <= 0.0) { discard; }\n"
"  let t_near = max(t0, 0.0);\n"
"\n"
"  let p = eo + rd * t_near;          // hit point, cell space\n"
"  let rho = length(p.xy);\n"
"\n"
"  // Surface normal from the biconcave profile rather than the spheroid, so\n"
"  // the dimple actually catches the light.\n"
"  let slope = profile_slope(min(rho, 0.999));\n"
"  let radial = select(vec2f(0.0, 0.0), p.xy / max(rho, 1e-5), rho > 1e-5);\n"
"  var n_local = normalize(vec3f(-slope * radial.x, -slope * radial.y, 1.0));\n"
"  if (p.z < 0.0) { n_local = vec3f(n_local.x, n_local.y, -n_local.z); }\n"
"  let n = normalize(in.bu * n_local.x + in.bv * n_local.y + in.bn * n_local.z);\n"
"\n"
"  // Path length through the cell, thinned at the centre by the dimple: this\n"
"  // is what makes a real one pale in the middle.\n"
"  let chord = max(t1 - t_near, 0.0);\n"
"  let thin = 1.0 - 0.62 * exp(-5.5 * rho * rho);\n"
"  let path = chord * thin;\n"
"\n"
"  // Beer-Lambert: transmitted light falls off with path length.\n"
"  let deep = vec3f(0.50, 0.04, 0.07);\n"
"  let pale = vec3f(0.95, 0.46, 0.44);\n"
"  var col = mix(deep, pale, exp(-5.5 * path));\n"
"\n"
"  let light = normalize(vec3f(-0.45, 0.72, 0.52));\n"
"  let view = -normalize(in.world - u.eye);\n"
"  let lambert = max(dot(n, light), 0.0);\n"
"  let half_v = normalize(light + view);\n"
"  let spec = pow(max(dot(n, half_v), 0.0), 28.0) * 0.30;\n"
"  col = col * (0.30 + 0.80 * lambert) + vec3f(spec);\n"
"\n"
"  // No two cells carry quite the same amount of haemoglobin.\n"
"  col *= 0.92 + 0.16 * in.seed;\n"
"\n"
"  // Depth comes from the actual surface, so cells intersect and occlude\n"
"  // each other correctly rather than by billboard distance.\n"
"  let hit_world = in.centre + (in.bu * p.x + in.bv * p.y + in.bn * p.z) * in.radius;\n"
"  let clip = u.view_proj * vec4f(hit_world, 1.0);\n"
"\n"
"  // The silhouette is an analytic curve, not a polygon edge, so coverage\n"
"  // comes from how fast the discriminant crosses zero at this pixel.\n"
"  let edge = sqrt(max(disc, 0.0));\n"
"  let alpha = clamp(edge / max(fwidth(edge), 1e-5), 0.0, 1.0);\n"
"\n"
"  var out: FSOut;\n"
"  out.color = vec4f(col, alpha);\n"
"  out.depth = clip.z / clip.w;\n"
"  return out;\n"
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

    WGPUVertexAttribute corner_attr = {};
    corner_attr.format = WGPUVertexFormat_Float32x2;
    corner_attr.offset = 0;
    corner_attr.shaderLocation = 0;

    std::array<WGPUVertexAttribute, 6> inst{};
    inst[0].format = WGPUVertexFormat_Float32x3;
    inst[0].offset = offsetof(Instance, px);
    inst[0].shaderLocation = 1;
    inst[1].format = WGPUVertexFormat_Float32;
    inst[1].offset = offsetof(Instance, radius);
    inst[1].shaderLocation = 2;
    inst[2].format = WGPUVertexFormat_Float32x3;
    inst[2].offset = offsetof(Instance, ux);
    inst[2].shaderLocation = 3;
    inst[3].format = WGPUVertexFormat_Float32;
    inst[3].offset = offsetof(Instance, seed);
    inst[3].shaderLocation = 4;
    inst[4].format = WGPUVertexFormat_Float32x3;
    inst[4].offset = offsetof(Instance, vx);
    inst[4].shaderLocation = 5;
    inst[5].format = WGPUVertexFormat_Uint32;
    inst[5].offset = offsetof(Instance, kind);
    inst[5].shaderLocation = 6;

    std::array<WGPUVertexBufferLayout, 2> layouts{};
    layouts[0].arrayStride    = sizeof(float) * 2;
    layouts[0].stepMode       = WGPUVertexStepMode_Vertex;
    layouts[0].attributeCount = 1;
    layouts[0].attributes     = &corner_attr;
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

    // Cells occlude each other, so depth is written as well as tested. The
    // antialiased rim is the one place that shows as a seam, which is why the
    // shader discards nearly-transparent pixels rather than blending them.
    WGPUDepthStencilState depth = {};
    depth.format              = depth_format;
    depth.depthWriteEnabled   = WGPUOptionalBool_True;
    depth.depthCompare        = WGPUCompareFunction_Less;
    depth.stencilFront.compare = WGPUCompareFunction_Always;
    depth.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor desc = {};
    desc.layout             = nullptr;  // auto layout, inferred from the shader
    desc.vertex.module      = module;
    desc.vertex.entryPoint  = str("vs");
    desc.vertex.bufferCount = layouts.size();
    desc.vertex.buffers     = layouts.data();
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode = WGPUCullMode_None;  // discs are seen from both sides
    desc.multisample.count  = 1;
    desc.multisample.mask   = 0xFFFFFFFFu;
    desc.depthStencil       = &depth;
    desc.fragment           = &fragment;

    g_pipeline = wgpuDeviceCreateRenderPipeline(g_device, &desc);
    wgpuShaderModuleRelease(module);
}

// Any two vectors perpendicular to the normal will do; pick the more stable
// of two candidate axes so the cross product never degenerates.
void basis_from_normal(const Vec3 &n, Vec3 &u, Vec3 &v) {
    const Vec3 axis = std::abs(n.y) < 0.9f ? Vec3{ 0.0f, 1.0f, 0.0f }
                                           : Vec3{ 1.0f, 0.0f, 0.0f };
    u = normalize(cross(axis, n));
    v = cross(n, u);
}

}  // namespace

void cells_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
                WGPUTextureFormat depth_format) {
    g_device = device;
    g_queue  = queue;

    create_pipeline(format, depth_format);
    g_instances.reserve(4096);

    constexpr std::array<float, 8> corners{ -1.0f, -1.0f,  1.0f, -1.0f,
                                             1.0f,  1.0f, -1.0f,  1.0f };
    WGPUBufferDescriptor cb = {};
    cb.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    cb.size  = sizeof(corners);
    g_corner_buffer = wgpuDeviceCreateBuffer(g_device, &cb);
    wgpuQueueWriteBuffer(g_queue, g_corner_buffer, 0, corners.data(), sizeof(corners));

    constexpr std::array<std::uint32_t, 6> indices{ 0, 1, 2, 2, 3, 0 };
    WGPUBufferDescriptor ib = {};
    ib.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
    ib.size  = sizeof(indices);
    g_index_buffer = wgpuDeviceCreateBuffer(g_device, &ib);
    wgpuQueueWriteBuffer(g_queue, g_index_buffer, 0, indices.data(), sizeof(indices));

    WGPUBufferDescriptor vb = {};
    vb.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    vb.size  = kMaxCells * sizeof(Instance);
    g_instance_buffer = wgpuDeviceCreateBuffer(g_device, &vb);

    WGPUBufferDescriptor ub = {};
    ub.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ub.size  = 112;  // mat4 (64) + eye, cam_right, cam_up (16 each)
    g_uniform_buffer = wgpuDeviceCreateBuffer(g_device, &ub);

    WGPUBindGroupEntry entry = {};
    entry.binding = 0;
    entry.buffer  = g_uniform_buffer;
    entry.size    = 112;

    WGPUBindGroupDescriptor bgd = {};
    bgd.layout     = wgpuRenderPipelineGetBindGroupLayout(g_pipeline, 0);
    bgd.entryCount = 1;
    bgd.entries    = &entry;
    g_bind_group = wgpuDeviceCreateBindGroup(g_device, &bgd);
}

void cells_draw(WGPURenderPassEncoder pass, const Mat4 &view_proj,
                const Vec3 &eye, const Vec3 &cam_right, const Vec3 &cam_up,
                std::span<const Cell> cells) {
    std::array<float, 28> uniforms{};
    for (std::size_t i = 0; i < 16; i++) uniforms[i] = view_proj[i];
    uniforms[16] = eye.x;       uniforms[17] = eye.y;       uniforms[18] = eye.z;
    uniforms[20] = cam_right.x; uniforms[21] = cam_right.y; uniforms[22] = cam_right.z;
    uniforms[24] = cam_up.x;    uniforms[25] = cam_up.y;    uniforms[26] = cam_up.z;
    wgpuQueueWriteBuffer(g_queue, g_uniform_buffer, 0, uniforms.data(),
                         sizeof(uniforms));

    const std::size_t count = cells.size() < kMaxCells ? cells.size() : kMaxCells;
    if (count == 0) return;

    g_instances.clear();
    for (std::size_t i = 0; i < count; i++) {
        const Cell &c = cells[i];
        Vec3 u, v;
        basis_from_normal(normalize(c.normal), u, v);

        // Spin within the disc plane, so cells are not all aligned.
        const float cs = std::cos(c.spin), sn = std::sin(c.spin);
        const Vec3 su = u * cs + v * sn;
        const Vec3 sv = v * cs - u * sn;

        g_instances.push_back(Instance{ c.pos.x, c.pos.y, c.pos.z, c.radius,
                                        su.x, su.y, su.z, c.seed,
                                        sv.x, sv.y, sv.z,
                                        static_cast<std::uint32_t>(c.kind) });
    }
    wgpuQueueWriteBuffer(g_queue, g_instance_buffer, 0, g_instances.data(),
                         g_instances.size() * sizeof(Instance));

    wgpuRenderPassEncoderSetPipeline(pass, g_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, g_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g_corner_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 1, g_instance_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(pass, g_index_buffer,
                                        WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);

    // Every cell is the same quad, so the whole frame is one instanced draw.
    wgpuRenderPassEncoderDrawIndexed(pass, 6, static_cast<std::uint32_t>(count), 0, 0, 0);
}
