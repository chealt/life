#include <array>
#include <cstddef>

#include "cells.h"

namespace {

constexpr std::size_t kMaxCells = 65536;

// What actually reaches the GPU. Kept separate from Cell so the public struct
// can grow fields the shader does not need.
struct Instance {
    float         x, y;
    float         radius;
    float         angle;
    float         squash;
    float         seed;
    std::uint32_t kind;
    float         _pad;
};
static_assert(sizeof(Instance) == 32, "instance stride is baked into the layout");

WGPUDevice         g_device;
WGPUQueue          g_queue;
WGPURenderPipeline g_pipeline;
WGPUBindGroup      g_bind_group;
WGPUBuffer         g_corner_buffer;
WGPUBuffer         g_index_buffer;
WGPUBuffer         g_instance_buffer;
WGPUBuffer         g_uniform_buffer;

std::array<Instance, kMaxCells> g_instances;

// A red blood cell is a biconcave disc: a torus of haemoglobin around a thin
// central dimple. Rather than model the geometry, the shader reconstructs the
// thickness of the cell at each pixel and treats colour as light passing
// through that much haemoglobin -- thick reads deep red, thin reads pale. That
// is why a real one has a pale centre, and it falls out for free here.
const char *kShader =
"struct Uniforms { screen: vec2f, _pad: vec2f };\n"
"@group(0) @binding(0) var<uniform> u: Uniforms;\n"
"\n"
"struct VSOut {\n"
"  @builtin(position) pos: vec4f,\n"
"  @location(0) local: vec2f,\n"
"  @location(1) @interpolate(flat) kind: u32,\n"
"  @location(2) @interpolate(flat) seed: f32,\n"
"};\n"
"\n"
"@vertex\n"
"fn vs(@location(0) corner: vec2f,\n"
"      @location(1) centre: vec2f,\n"
"      @location(2) radius: f32,\n"
"      @location(3) angle: f32,\n"
"      @location(4) squash: f32,\n"
"      @location(5) seed: f32,\n"
"      @location(6) kind: u32) -> VSOut {\n"
"  // Squash first, then rotate: an ellipse standing in for a disc tilted\n"
"  // away from the viewer.\n"
"  let e = vec2f(corner.x, corner.y * squash);\n"
"  let c = cos(angle);\n"
"  let s = sin(angle);\n"
"  let r = vec2f(e.x * c - e.y * s, e.x * s + e.y * c);\n"
"  let p = centre + r * radius;\n"
"\n"
"  var out: VSOut;\n"
"  out.pos = vec4f(p.x / u.screen.x * 2.0 - 1.0,\n"
"                  1.0 - p.y / u.screen.y * 2.0, 0.0, 1.0);\n"
"  out.local = corner;   // unit disc space, so shading stays circular\n"
"  out.kind = kind;\n"
"  out.seed = seed;\n"
"  return out;\n"
"}\n"
"\n"
"fn red_blood(local: vec2f, seed: f32) -> vec4f {\n"
"  let d = length(local);\n"
"  // fwidth gives one pixel in local units, so the edge stays one pixel wide\n"
"  // at any radius or devicePixelRatio.\n"
"  let aa = fwidth(d);\n"
"  let alpha = 1.0 - smoothstep(1.0 - aa, 1.0, d);\n"
"  if (alpha <= 0.0) { discard; }\n"
"\n"
"  // Thickness through the cell: a rounded disc minus a central depression.\n"
"  let disc   = sqrt(max(1.0 - d * d, 0.0));\n"
"  let dimple = 0.62 * exp(-5.5 * d * d);\n"
"  let thick  = max(disc - dimple, 0.0);\n"
"\n"
"  // Beer-Lambert: transmitted light falls off with path length.\n"
"  let deep = vec3f(0.52, 0.045, 0.075);\n"
"  let pale = vec3f(0.93, 0.42, 0.40);\n"
"  var col = mix(deep, pale, exp(-2.6 * thick));\n"
"\n"
"  // Thin dark outline where the membrane turns away from the viewer.\n"
"  let edge = smoothstep(0.86, 1.0, d);\n"
"  col = mix(col, deep * 0.7, edge * 0.7);\n"
"\n"
"  // A soft highlight on the raised torus, lit from the upper left.\n"
"  let dir   = local / max(d, 1e-4);\n"
"  let ndl   = max(dot(dir, normalize(vec2f(-0.5, -0.7))), 0.0);\n"
"  let ridge = smoothstep(0.30, 0.72, d) * (1.0 - smoothstep(0.78, 1.0, d));\n"
"  col += vec3f(0.20, 0.10, 0.10) * ndl * ridge;\n"
"\n"
"  // No two cells carry quite the same amount of haemoglobin.\n"
"  col *= 0.92 + 0.16 * seed;\n"
"  return vec4f(col, alpha);\n"
"}\n"
"\n"
"@fragment\n"
"fn fs(in: VSOut) -> @location(0) vec4f {\n"
"  return red_blood(in.local, in.seed);\n"
"}\n";

WGPUStringView str(const char *s) {
    WGPUStringView v;
    v.data = s;
    v.length = WGPU_STRLEN;
    return v;
}

void create_pipeline(WGPUTextureFormat format) {
    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = str(kShader);
    WGPUShaderModuleDescriptor module_desc = {};
    module_desc.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_device, &module_desc);

    // Buffer 0 is the unit quad, stepped per vertex; buffer 1 is per instance.
    WGPUVertexAttribute corner_attr = {};
    corner_attr.format = WGPUVertexFormat_Float32x2;
    corner_attr.offset = 0;
    corner_attr.shaderLocation = 0;

    std::array<WGPUVertexAttribute, 6> inst{};
    inst[0].format = WGPUVertexFormat_Float32x2;
    inst[0].offset = offsetof(Instance, x);
    inst[0].shaderLocation = 1;
    inst[1].format = WGPUVertexFormat_Float32;
    inst[1].offset = offsetof(Instance, radius);
    inst[1].shaderLocation = 2;
    inst[2].format = WGPUVertexFormat_Float32;
    inst[2].offset = offsetof(Instance, angle);
    inst[2].shaderLocation = 3;
    inst[3].format = WGPUVertexFormat_Float32;
    inst[3].offset = offsetof(Instance, squash);
    inst[3].shaderLocation = 4;
    inst[4].format = WGPUVertexFormat_Float32;
    inst[4].offset = offsetof(Instance, seed);
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

    // Straight (non-premultiplied) alpha, matching the UI renderer.
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

    WGPURenderPipelineDescriptor desc = {};
    desc.layout             = nullptr;  // auto layout, inferred from the shader
    desc.vertex.module      = module;
    desc.vertex.entryPoint  = str("vs");
    desc.vertex.bufferCount = layouts.size();
    desc.vertex.buffers     = layouts.data();
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.multisample.count  = 1;
    desc.multisample.mask   = 0xFFFFFFFFu;
    desc.fragment           = &fragment;

    g_pipeline = wgpuDeviceCreateRenderPipeline(g_device, &desc);
    wgpuShaderModuleRelease(module);
}

}  // namespace

void cells_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format) {
    g_device = device;
    g_queue  = queue;

    create_pipeline(format);

    // The quad and its indices never change, so both are written once here.
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
    vb.size  = sizeof(g_instances);
    g_instance_buffer = wgpuDeviceCreateBuffer(g_device, &vb);

    WGPUBufferDescriptor ub = {};
    ub.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ub.size  = 16;
    g_uniform_buffer = wgpuDeviceCreateBuffer(g_device, &ub);

    WGPUBindGroupEntry entry = {};
    entry.binding = 0;
    entry.buffer  = g_uniform_buffer;
    entry.size    = 16;

    WGPUBindGroupDescriptor bgd = {};
    bgd.layout     = wgpuRenderPipelineGetBindGroupLayout(g_pipeline, 0);
    bgd.entryCount = 1;
    bgd.entries    = &entry;
    g_bind_group = wgpuDeviceCreateBindGroup(g_device, &bgd);
}

void cells_draw(WGPURenderPassEncoder pass, int width, int height,
                std::span<const Cell> cells) {
    const std::array<float, 4> uniforms{ static_cast<float>(width),
                                         static_cast<float>(height), 0.0f, 0.0f };
    wgpuQueueWriteBuffer(g_queue, g_uniform_buffer, 0, uniforms.data(),
                         sizeof(uniforms));

    const std::size_t count = cells.size() < kMaxCells ? cells.size() : kMaxCells;
    if (count == 0) return;

    for (std::size_t i = 0; i < count; i++) {
        const Cell &c = cells[i];
        g_instances[i] = Instance{ c.x, c.y, c.radius, c.angle, c.squash,
                                   c.seed, static_cast<std::uint32_t>(c.kind), 0.0f };
    }
    wgpuQueueWriteBuffer(g_queue, g_instance_buffer, 0, g_instances.data(),
                         count * sizeof(Instance));

    wgpuRenderPassEncoderSetPipeline(pass, g_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, g_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g_corner_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 1, g_instance_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(pass, g_index_buffer,
                                        WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);

    // Every cell is the same quad, so the whole frame is one instanced draw.
    wgpuRenderPassEncoderDrawIndexed(pass, 6, static_cast<std::uint32_t>(count), 0, 0, 0);
}
