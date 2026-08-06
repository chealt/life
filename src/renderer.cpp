#include <string.h>
#include <stdlib.h>
#include <stddef.h>

#include "renderer.h"

// microui itself is gone; its atlas is kept purely as font data, and that file
// is written against microui's types, so the header still has to be visible.
extern "C" {
#include "microui.h"
}

// microui's atlas indexes its initialiser with [MU_ICON_CLOSE] = {...}, a C99
// array designator that C++ never adopted. Clang accepts it as an extension;
// the warning is upstream's to fix, not ours.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-designator"
#include "atlas.inl"
#pragma clang diagnostic pop

#define MAX_QUADS   8192
#define MAX_BATCHES 256

typedef struct {
    float    x, y;
    float    u, v;
    uint32_t color;  // RGBA8, unorm
} Vertex;

// A run of quads sharing one scissor rect. microui interleaves clip commands
// with geometry, so we close a batch whenever the clip changes.
typedef struct {
    uint32_t first_index;
    uint32_t index_count;
    UiRect   clip;
} Batch;

static WGPUDevice          g_device;
static WGPUQueue           g_queue;
static WGPURenderPipeline  g_pipeline;
static WGPUBindGroup       g_bind_group;
static WGPUBuffer          g_vertex_buffer;
static WGPUBuffer          g_index_buffer;
static WGPUBuffer          g_uniform_buffer;

static Vertex g_vertices[MAX_QUADS * 4];
static Batch  g_batches[MAX_BATCHES];
static int    g_quad_count;
static int    g_batch_count;
static UiRect g_clip;
static int    g_width, g_height, g_scale = 1;

static const char *kShader =
"struct Uniforms { screen: vec2f, _pad: vec2f };\n"
"@group(0) @binding(0) var<uniform> u: Uniforms;\n"
"@group(0) @binding(1) var samp: sampler;\n"
"@group(0) @binding(2) var tex: texture_2d<f32>;\n"
"\n"
"struct VSOut {\n"
"  @builtin(position) pos: vec4f,\n"
"  @location(0) uv: vec2f,\n"
"  @location(1) color: vec4f,\n"
"};\n"
"\n"
"@vertex\n"
"fn vs(@location(0) pos: vec2f, @location(1) uv: vec2f,\n"
"      @location(2) color: vec4f) -> VSOut {\n"
"  var out: VSOut;\n"
"  out.pos = vec4f(pos.x / u.screen.x * 2.0 - 1.0,\n"
"                  1.0 - pos.y / u.screen.y * 2.0, 0.0, 1.0);\n"
"  out.uv = uv;\n"
"  out.color = color;\n"
"  return out;\n"
"}\n"
"\n"
"@fragment\n"
"fn fs(in: VSOut) -> @location(0) vec4f {\n"
"  // The atlas is coverage-only; colour comes from the vertex.\n"
"  return vec4f(in.color.rgb, in.color.a * textureSample(tex, samp, in.uv).r);\n"
"}\n";

static WGPUStringView str(const char *s) {
    WGPUStringView v;
    v.data = s;
    v.length = WGPU_STRLEN;
    return v;
}

static int rect_eq(UiRect a, UiRect b) {
    return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h;
}

static WGPUTexture create_atlas_texture(void) {
    WGPUTextureDescriptor desc = {};
    desc.dimension     = WGPUTextureDimension_2D;
    desc.size.width    = ATLAS_WIDTH;
    desc.size.height   = ATLAS_HEIGHT;
    desc.size.depthOrArrayLayers = 1;
    desc.format        = WGPUTextureFormat_R8Unorm;
    desc.mipLevelCount = 1;
    desc.sampleCount   = 1;
    desc.usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    WGPUTexture texture = wgpuDeviceCreateTexture(g_device, &desc);

    WGPUTexelCopyTextureInfo dst = {};
    dst.texture  = texture;
    dst.aspect   = WGPUTextureAspect_All;

    WGPUTexelCopyBufferLayout layout = {};
    layout.bytesPerRow  = ATLAS_WIDTH;
    layout.rowsPerImage = ATLAS_HEIGHT;

    WGPUExtent3D extent = { ATLAS_WIDTH, ATLAS_HEIGHT, 1 };
    wgpuQueueWriteTexture(g_queue, &dst, atlas_texture,
                          sizeof(atlas_texture), &layout, &extent);
    return texture;
}

static void create_pipeline(WGPUTextureFormat format, WGPUTextureFormat depth_format) {
    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = str(kShader);
    WGPUShaderModuleDescriptor module_desc = {};
    module_desc.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_device, &module_desc);

    WGPUVertexAttribute attrs[3] = {};
    attrs[0].format = WGPUVertexFormat_Float32x2;
    attrs[0].offset = offsetof(Vertex, x);
    attrs[0].shaderLocation = 0;
    attrs[1].format = WGPUVertexFormat_Float32x2;
    attrs[1].offset = offsetof(Vertex, u);
    attrs[1].shaderLocation = 1;
    attrs[2].format = WGPUVertexFormat_Unorm8x4;
    attrs[2].offset = offsetof(Vertex, color);
    attrs[2].shaderLocation = 2;

    WGPUVertexBufferLayout vb_layout = {};
    vb_layout.arrayStride    = sizeof(Vertex);
    vb_layout.stepMode       = WGPUVertexStepMode_Vertex;
    vb_layout.attributeCount = 3;
    vb_layout.attributes     = attrs;

    // Straight (non-premultiplied) alpha, matching microui's colours.
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

    // The render pass has a depth attachment, so this pipeline has to declare
    // one too -- but the UI is an overlay and neither tests nor writes it.
    WGPUDepthStencilState depth = {};
    depth.format               = depth_format;
    depth.depthWriteEnabled    = WGPUOptionalBool_False;
    depth.depthCompare         = WGPUCompareFunction_Always;
    depth.stencilFront.compare = WGPUCompareFunction_Always;
    depth.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor desc = {};
    desc.depthStencil           = &depth;
    desc.layout                 = NULL;  // auto layout, inferred from the shader
    desc.vertex.module          = module;
    desc.vertex.entryPoint      = str("vs");
    desc.vertex.bufferCount     = 1;
    desc.vertex.buffers         = &vb_layout;
    desc.primitive.topology     = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode     = WGPUCullMode_None;
    desc.multisample.count      = 1;
    desc.multisample.mask       = 0xFFFFFFFFu;
    desc.fragment               = &fragment;

    g_pipeline = wgpuDeviceCreateRenderPipeline(g_device, &desc);
    wgpuShaderModuleRelease(module);
}

void r_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
            WGPUTextureFormat depth_format) {
    g_device = device;
    g_queue  = queue;

    create_pipeline(format, depth_format);

    WGPUBufferDescriptor vb = {};
    vb.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    vb.size  = sizeof(g_vertices);
    g_vertex_buffer = wgpuDeviceCreateBuffer(g_device, &vb);

    // Every primitive is a quad, so the index buffer is fixed: build it once.
    uint32_t *indices = (uint32_t *)malloc(MAX_QUADS * 6 * sizeof(uint32_t));
    for (int i = 0; i < MAX_QUADS; i++) {
        uint32_t v = (uint32_t)i * 4;
        uint32_t *p = &indices[i * 6];
        p[0] = v; p[1] = v + 1; p[2] = v + 2;
        p[3] = v + 2; p[4] = v + 3; p[5] = v;
    }
    WGPUBufferDescriptor ib = {};
    ib.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
    ib.size  = MAX_QUADS * 6 * sizeof(uint32_t);
    g_index_buffer = wgpuDeviceCreateBuffer(g_device, &ib);
    wgpuQueueWriteBuffer(g_queue, g_index_buffer, 0, indices, ib.size);
    free(indices);

    WGPUBufferDescriptor ub = {};
    ub.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ub.size  = 16;
    g_uniform_buffer = wgpuDeviceCreateBuffer(g_device, &ub);

    WGPUTexture atlas_tex = create_atlas_texture();
    WGPUTextureView atlas_view = wgpuTextureCreateView(atlas_tex, NULL);

    // Nearest filtering keeps the bitmap font crisp under integer UI scaling.
    WGPUSamplerDescriptor sd = {};
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter    = WGPUFilterMode_Nearest;
    sd.minFilter    = WGPUFilterMode_Nearest;
    sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sd.maxAnisotropy = 1;
    WGPUSampler sampler = wgpuDeviceCreateSampler(g_device, &sd);

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0;
    entries[0].buffer  = g_uniform_buffer;
    entries[0].size    = 16;
    entries[1].binding = 1;
    entries[1].sampler = sampler;
    entries[2].binding = 2;
    entries[2].textureView = atlas_view;

    WGPUBindGroupDescriptor bgd = {};
    bgd.layout     = wgpuRenderPipelineGetBindGroupLayout(g_pipeline, 0);
    bgd.entryCount = 3;
    bgd.entries    = entries;
    g_bind_group = wgpuDeviceCreateBindGroup(g_device, &bgd);
}

static void push_quad(UiRect dst, mu_Rect src, UiColor color) {
    if (g_quad_count >= MAX_QUADS) return;

    Batch *b = g_batch_count > 0 ? &g_batches[g_batch_count - 1] : NULL;
    if (b == NULL || !rect_eq(b->clip, g_clip)) {
        if (g_batch_count >= MAX_BATCHES) return;
        b = &g_batches[g_batch_count++];
        b->first_index = (uint32_t)g_quad_count * 6;
        b->index_count = 0;
        b->clip = g_clip;
    }

    // The atlas dimensions are unscoped enum constants; C++23 deprecates using
    // those directly in float arithmetic, so name them as floats first.
    constexpr float atlas_w = static_cast<float>(ATLAS_WIDTH);
    constexpr float atlas_h = static_cast<float>(ATLAS_HEIGHT);
    const float u0 = (float)src.x / atlas_w;
    const float v0 = (float)src.y / atlas_h;
    const float u1 = (float)(src.x + src.w) / atlas_w;
    const float v1 = (float)(src.y + src.h) / atlas_h;
    const float x0 = (float)dst.x, y0 = (float)dst.y;
    const float x1 = (float)(dst.x + dst.w), y1 = (float)(dst.y + dst.h);
    const uint32_t rgba = (uint32_t)color.r | ((uint32_t)color.g << 8) |
                          ((uint32_t)color.b << 16) | ((uint32_t)color.a << 24);

    Vertex *v = &g_vertices[g_quad_count * 4];
    v[0] = Vertex{ x0, y0, u0, v0, rgba };
    v[1] = Vertex{ x1, y0, u1, v0, rgba };
    v[2] = Vertex{ x1, y1, u1, v1, rgba };
    v[3] = Vertex{ x0, y1, u0, v1, rgba };

    b->index_count += 6;
    g_quad_count++;
}

void r_begin(int width, int height, int scale) {
    g_width  = width;
    g_height = height;
    g_scale  = scale < 1 ? 1 : scale;
    g_quad_count  = 0;
    g_batch_count = 0;
    g_clip = UiRect{ 0, 0, width, height };

    float uniforms[4] = { (float)width, (float)height, 0.0f, 0.0f };
    wgpuQueueWriteBuffer(g_queue, g_uniform_buffer, 0, uniforms, sizeof(uniforms));
}

void r_draw_rect(UiRect rect, UiColor color) {
    push_quad(rect, atlas[ATLAS_WHITE], color);
}

void r_draw_text(const char *text, int x, int y, UiColor color) {
    UiRect dst{ x, y, 0, 0 };
    for (const char *p = text; *p; p++) {
        if ((*p & 0xc0) == 0x80) continue;  // UTF-8 continuation byte
        int chr = (unsigned char)*p;
        if (chr > 127) chr = 127;
        mu_Rect src = atlas[ATLAS_FONT + chr];
        dst.w = src.w;
        dst.h = src.h;
        push_quad(dst, src, color);
        dst.x += dst.w;
    }
}


void r_set_clip_rect(UiRect rect) {
    g_clip = rect;
}

void r_end(WGPURenderPassEncoder pass) {
    if (g_quad_count == 0) return;

    wgpuQueueWriteBuffer(g_queue, g_vertex_buffer, 0, g_vertices,
                         (size_t)g_quad_count * 4 * sizeof(Vertex));

    wgpuRenderPassEncoderSetPipeline(pass, g_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, g_bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g_vertex_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(pass, g_index_buffer,
                                        WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);

    for (int i = 0; i < g_batch_count; i++) {
        Batch *b = &g_batches[i];
        if (b->index_count == 0) continue;

        // microui's unclipped rect is huge; WebGPU rejects a scissor that
        // leaves the attachment, so clamp it to the framebuffer.
        int x0 = b->clip.x < 0 ? 0 : b->clip.x;
        int y0 = b->clip.y < 0 ? 0 : b->clip.y;
        int x1 = b->clip.x + b->clip.w;
        int y1 = b->clip.y + b->clip.h;
        if (x1 > g_width)  x1 = g_width;
        if (y1 > g_height) y1 = g_height;
        if (x1 <= x0 || y1 <= y0) continue;

        // Scissor is in framebuffer pixels, not the logical units above.
        x0 *= g_scale; y0 *= g_scale; x1 *= g_scale; y1 *= g_scale;
        wgpuRenderPassEncoderSetScissorRect(pass, (uint32_t)x0, (uint32_t)y0,
                                            (uint32_t)(x1 - x0), (uint32_t)(y1 - y0));
        wgpuRenderPassEncoderDrawIndexed(pass, b->index_count, 1, b->first_index, 0, 0);
    }
}

int r_text_width(const char *text, int len) {
    if (len < 0) len = (int)strlen(text);
    int res = 0;
    for (const char *p = text; *p && len--; p++) {
        if ((*p & 0xc0) == 0x80) continue;
        int chr = (unsigned char)*p;
        if (chr > 127) chr = 127;
        res += atlas[ATLAS_FONT + chr].w;
    }
    return res;
}

int r_text_height() {
    return 18;
}
