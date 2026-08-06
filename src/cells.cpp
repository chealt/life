#include <stddef.h>

#include "cells.h"

#define MAX_CELLS 4096

// One instance per cell. 16 bytes, no padding needed.
typedef struct {
    float    x, y;    // centre, logical units
    float    radius;  // logical units
    uint32_t color;   // RGBA8, unorm
} Instance;

static WGPUDevice         g_device;
static WGPUQueue          g_queue;
static WGPURenderPipeline g_pipeline;
static WGPUBindGroup      g_bind_group;
static WGPUBuffer         g_corner_buffer;
static WGPUBuffer         g_index_buffer;
static WGPUBuffer         g_instance_buffer;
static WGPUBuffer         g_uniform_buffer;

static Instance g_instances[MAX_CELLS];
static int      g_count;

// A red blood cell is a biconcave disc: dense red at the rim, pale in the
// middle where it is thinnest. Cheaper to shade that from the distance to the
// centre than to model the geometry.
static const char *kShader =
"struct Uniforms { screen: vec2f, _pad: vec2f };\n"
"@group(0) @binding(0) var<uniform> u: Uniforms;\n"
"\n"
"struct VSOut {\n"
"  @builtin(position) pos: vec4f,\n"
"  @location(0) local: vec2f,\n"
"  @location(1) tint: vec4f,\n"
"};\n"
"\n"
"@vertex\n"
"fn vs(@location(0) corner: vec2f,\n"
"      @location(1) centre: vec2f,\n"
"      @location(2) radius: f32,\n"
"      @location(3) tint: vec4f) -> VSOut {\n"
"  let p = centre + corner * radius;\n"
"  var out: VSOut;\n"
"  out.pos = vec4f(p.x / u.screen.x * 2.0 - 1.0,\n"
"                  1.0 - p.y / u.screen.y * 2.0, 0.0, 1.0);\n"
"  out.local = corner;\n"
"  out.tint = tint;\n"
"  return out;\n"
"}\n"
"\n"
"@fragment\n"
"fn fs(in: VSOut) -> @location(0) vec4f {\n"
"  let d = length(in.local);\n"
"  // fwidth gives one pixel in local units, so the edge stays one pixel wide\n"
"  // at any radius or devicePixelRatio.\n"
"  let aa = fwidth(d);\n"
"  let alpha = 1.0 - smoothstep(1.0 - aa, 1.0, d);\n"
"  if (alpha <= 0.0) { discard; }\n"
"\n"
"  let base = in.tint.rgb;\n"
"  let pale = clamp(base * 1.75, vec3f(0.0), vec3f(1.0));\n"
"  let dip  = 1.0 - smoothstep(0.0, 0.5, d);   // 1 at the centre\n"
"  var col  = mix(base, pale, dip * 0.85);\n"
"  col = col * mix(1.0, 0.72, smoothstep(0.8, 1.0, d));  // darken the rim\n"
"  return vec4f(col, in.tint.a * alpha);\n"
"}\n";

static WGPUStringView str(const char *s) {
    WGPUStringView v;
    v.data = s;
    v.length = WGPU_STRLEN;
    return v;
}

static void create_pipeline(WGPUTextureFormat format) {
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

    WGPUVertexAttribute inst_attrs[3] = {};
    inst_attrs[0].format = WGPUVertexFormat_Float32x2;
    inst_attrs[0].offset = offsetof(Instance, x);
    inst_attrs[0].shaderLocation = 1;
    inst_attrs[1].format = WGPUVertexFormat_Float32;
    inst_attrs[1].offset = offsetof(Instance, radius);
    inst_attrs[1].shaderLocation = 2;
    inst_attrs[2].format = WGPUVertexFormat_Unorm8x4;
    inst_attrs[2].offset = offsetof(Instance, color);
    inst_attrs[2].shaderLocation = 3;

    WGPUVertexBufferLayout layouts[2] = {};
    layouts[0].arrayStride    = sizeof(float) * 2;
    layouts[0].stepMode       = WGPUVertexStepMode_Vertex;
    layouts[0].attributeCount = 1;
    layouts[0].attributes     = &corner_attr;
    layouts[1].arrayStride    = sizeof(Instance);
    layouts[1].stepMode       = WGPUVertexStepMode_Instance;
    layouts[1].attributeCount = 3;
    layouts[1].attributes     = inst_attrs;

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
    desc.layout             = NULL;  // auto layout, inferred from the shader
    desc.vertex.module      = module;
    desc.vertex.entryPoint  = str("vs");
    desc.vertex.bufferCount = 2;
    desc.vertex.buffers     = layouts;
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.multisample.count  = 1;
    desc.multisample.mask   = 0xFFFFFFFFu;
    desc.fragment           = &fragment;

    g_pipeline = wgpuDeviceCreateRenderPipeline(g_device, &desc);
    wgpuShaderModuleRelease(module);
}

void cells_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format) {
    g_device = device;
    g_queue  = queue;

    create_pipeline(format);

    // The quad and its indices never change, so both are written once here.
    const float corners[8] = { -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, 1.0f,  -1.0f, 1.0f };
    WGPUBufferDescriptor cb = {};
    cb.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    cb.size  = sizeof(corners);
    g_corner_buffer = wgpuDeviceCreateBuffer(g_device, &cb);
    wgpuQueueWriteBuffer(g_queue, g_corner_buffer, 0, corners, sizeof(corners));

    const uint32_t indices[6] = { 0, 1, 2, 2, 3, 0 };
    WGPUBufferDescriptor ib = {};
    ib.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
    ib.size  = sizeof(indices);
    g_index_buffer = wgpuDeviceCreateBuffer(g_device, &ib);
    wgpuQueueWriteBuffer(g_queue, g_index_buffer, 0, indices, sizeof(indices));

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

void cells_begin(int width, int height) {
    g_count = 0;
    float uniforms[4] = { (float)width, (float)height, 0.0f, 0.0f };
    wgpuQueueWriteBuffer(g_queue, g_uniform_buffer, 0, uniforms, sizeof(uniforms));
}

void cells_add(float x, float y, float radius, uint32_t rgba) {
    if (g_count >= MAX_CELLS) return;
    g_instances[g_count++] = Instance{ x, y, radius, rgba };
}

void cells_end(WGPURenderPassEncoder pass) {
    if (g_count == 0) return;

    wgpuQueueWriteBuffer(g_queue, g_instance_buffer, 0, g_instances,
                         (size_t)g_count * sizeof(Instance));

    wgpuRenderPassEncoderSetPipeline(pass, g_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, g_bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g_corner_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 1, g_instance_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(pass, g_index_buffer,
                                        WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);

    // Every cell is the same quad, so the whole frame is one instanced draw.
    wgpuRenderPassEncoderDrawIndexed(pass, 6, (uint32_t)g_count, 0, 0, 0);
}
