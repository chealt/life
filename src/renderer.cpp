#include <cstring>
#include <cstddef>
#include <cstdlib>

#include "renderer.h"
#include "font_atlas.h"

#define MAX_QUADS   8192
#define MAX_BATCHES 256

// Text is laid out at this height in logical units. The atlas is a distance
// field, so this is a free choice rather than the size it was baked at.
static constexpr float kTextSize = 15.0f;
static constexpr float kTextScale = kTextSize / kFontPixelSize;

struct Vertex {
    float         x, y;
    float         u, v;
    std::uint32_t color;  // RGBA8, unorm
};

// A run of quads sharing one scissor rect; a batch closes whenever the clip
// changes.
struct Batch {
    std::uint32_t first_index;
    std::uint32_t index_count;
    UiRect        clip;
};

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

// Codepoint to glyph index. Only Latin-1 is covered, which is all the atlas
// holds.
static std::int16_t g_glyph_index[256];

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
"  // The atlas stores distance from the outline, with 0.5 sitting exactly on\n"
"  // it. Thresholding against the screen-space rate of change antialiases the\n"
"  // edge at whatever size the glyph happens to be drawn.\n"
"  let d = textureSample(tex, samp, in.uv).r;\n"
"  let w = max(fwidth(d), 1e-4);\n"
"  let coverage = smoothstep(0.5 - w, 0.5 + w, d);\n"
"  return vec4f(in.color.rgb, in.color.a * coverage);\n"
"}\n";

static WGPUStringView str(const char *s) {
    WGPUStringView v;
    v.data = s;
    v.length = WGPU_STRLEN;
    return v;
}

static bool rect_eq(UiRect a, UiRect b) {
    return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h;
}

static WGPUTexture create_atlas_texture() {
    WGPUTextureDescriptor desc = {};
    desc.dimension     = WGPUTextureDimension_2D;
    desc.size.width    = kFontAtlasW;
    desc.size.height   = kFontAtlasH;
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
    layout.bytesPerRow  = kFontAtlasW;
    layout.rowsPerImage = kFontAtlasH;

    WGPUExtent3D extent = { kFontAtlasW, kFontAtlasH, 1 };
    wgpuQueueWriteTexture(g_queue, &dst, kFontAtlas, sizeof(kFontAtlas),
                          &layout, &extent);
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

    // Straight (non-premultiplied) alpha.
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
    desc.depthStencil       = &depth;
    desc.layout             = nullptr;  // auto layout, inferred from the shader
    desc.vertex.module      = module;
    desc.vertex.entryPoint  = str("vs");
    desc.vertex.bufferCount = 1;
    desc.vertex.buffers     = &vb_layout;
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.multisample.count  = 1;
    desc.multisample.mask   = 0xFFFFFFFFu;
    desc.fragment           = &fragment;

    g_pipeline = wgpuDeviceCreateRenderPipeline(g_device, &desc);
    wgpuShaderModuleRelease(module);
}

void r_init(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
            WGPUTextureFormat depth_format) {
    g_device = device;
    g_queue  = queue;

    for (int i = 0; i < 256; i++) g_glyph_index[i] = -1;
    for (int i = 0; i < kFontGlyphCount; i++) {
        const unsigned cp = kFontGlyphs[i].codepoint;
        if (cp < 256) g_glyph_index[cp] = static_cast<std::int16_t>(i);
    }

    create_pipeline(format, depth_format);

    WGPUBufferDescriptor vb = {};
    vb.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    vb.size  = sizeof(g_vertices);
    g_vertex_buffer = wgpuDeviceCreateBuffer(g_device, &vb);

    // Every primitive is a quad, so the index buffer is fixed: build it once.
    std::uint32_t *indices =
        (std::uint32_t *)malloc(MAX_QUADS * 6 * sizeof(std::uint32_t));
    for (int i = 0; i < MAX_QUADS; i++) {
        std::uint32_t v = (std::uint32_t)i * 4;
        std::uint32_t *p = &indices[i * 6];
        p[0] = v; p[1] = v + 1; p[2] = v + 2;
        p[3] = v + 2; p[4] = v + 3; p[5] = v;
    }
    WGPUBufferDescriptor ib = {};
    ib.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
    ib.size  = MAX_QUADS * 6 * sizeof(std::uint32_t);
    g_index_buffer = wgpuDeviceCreateBuffer(g_device, &ib);
    wgpuQueueWriteBuffer(g_queue, g_index_buffer, 0, indices, ib.size);
    free(indices);

    WGPUBufferDescriptor ub = {};
    ub.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ub.size  = 16;
    g_uniform_buffer = wgpuDeviceCreateBuffer(g_device, &ub);

    WGPUTexture atlas_tex = create_atlas_texture();
    WGPUTextureView atlas_view = wgpuTextureCreateView(atlas_tex, nullptr);

    // Linear, unlike the old bitmap font: a distance field has to be
    // interpolated for the reconstructed outline to come out smooth.
    WGPUSamplerDescriptor sd = {};
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter    = WGPUFilterMode_Linear;
    sd.minFilter    = WGPUFilterMode_Linear;
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

static void push_quad(float x0, float y0, float x1, float y1,
                      float u0, float v0, float u1, float v1, UiColor color) {
    if (g_quad_count >= MAX_QUADS) return;

    Batch *b = g_batch_count > 0 ? &g_batches[g_batch_count - 1] : nullptr;
    if (b == nullptr || !rect_eq(b->clip, g_clip)) {
        if (g_batch_count >= MAX_BATCHES) return;
        b = &g_batches[g_batch_count++];
        b->first_index = (std::uint32_t)g_quad_count * 6;
        b->index_count = 0;
        b->clip = g_clip;
    }

    const std::uint32_t rgba = (std::uint32_t)color.r |
                               ((std::uint32_t)color.g << 8) |
                               ((std::uint32_t)color.b << 16) |
                               ((std::uint32_t)color.a << 24);

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
    // Glyph 0 is a solid block. Sampling its middle keeps the filtered edge
    // of the block out of the result.
    const FontGlyph &blk = kFontGlyphs[0];
    const float u = ((float)blk.x + (float)blk.w * 0.5f) / kFontAtlasW;
    const float v = ((float)blk.y + (float)blk.h * 0.5f) / kFontAtlasH;

    push_quad((float)rect.x, (float)rect.y,
              (float)(rect.x + rect.w), (float)(rect.y + rect.h),
              u, v, u, v, color);
}

// Decodes one codepoint, advancing `p`. Only the one- and two-byte forms are
// handled, which covers everything in the atlas.
static unsigned next_codepoint(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    unsigned cp = *s++;
    if ((cp & 0xE0) == 0xC0 && (*s & 0xC0) == 0x80) {
        cp = ((cp & 0x1F) << 6) | (*s++ & 0x3F);
    } else if (cp >= 0x80) {
        // Anything longer is not in the atlas; skip its continuation bytes.
        while ((*s & 0xC0) == 0x80) s++;
        cp = '?';
    }
    *p = (const char *)s;
    return cp;
}

void r_draw_text(const char *text, int x, int y, UiColor color) {
    float pen = (float)x;
    const float baseline = (float)y + kFontAscent * kTextScale;

    for (const char *p = text; *p; ) {
        const unsigned cp = next_codepoint(&p);
        const std::int16_t gi = cp < 256 ? g_glyph_index[cp] : -1;
        if (gi < 0) continue;

        const FontGlyph &g = kFontGlyphs[gi];
        if (g.w > 0) {
            const float x0 = pen + g.xoff * kTextScale;
            const float y0 = baseline + g.yoff * kTextScale;
            push_quad(x0, y0,
                      x0 + (float)g.w * kTextScale, y0 + (float)g.h * kTextScale,
                      (float)g.x / kFontAtlasW, (float)g.y / kFontAtlasH,
                      (float)(g.x + g.w) / kFontAtlasW,
                      (float)(g.y + g.h) / kFontAtlasH, color);
        }
        pen += g.advance * kTextScale;
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
    wgpuRenderPassEncoderSetBindGroup(pass, 0, g_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g_vertex_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(pass, g_index_buffer,
                                        WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);

    for (int i = 0; i < g_batch_count; i++) {
        Batch *b = &g_batches[i];
        if (b->index_count == 0) continue;

        // WebGPU rejects a scissor that leaves the attachment, so clamp it to
        // the framebuffer.
        int x0 = b->clip.x < 0 ? 0 : b->clip.x;
        int y0 = b->clip.y < 0 ? 0 : b->clip.y;
        int x1 = b->clip.x + b->clip.w;
        int y1 = b->clip.y + b->clip.h;
        if (x1 > g_width)  x1 = g_width;
        if (y1 > g_height) y1 = g_height;
        if (x1 <= x0 || y1 <= y0) continue;

        // Scissor is in framebuffer pixels, not the logical units above.
        x0 *= g_scale; y0 *= g_scale; x1 *= g_scale; y1 *= g_scale;
        wgpuRenderPassEncoderSetScissorRect(pass, (std::uint32_t)x0, (std::uint32_t)y0,
                                            (std::uint32_t)(x1 - x0),
                                            (std::uint32_t)(y1 - y0));
        wgpuRenderPassEncoderDrawIndexed(pass, b->index_count, 1, b->first_index, 0, 0);
    }
}

int r_text_width(const char *text, int len) {
    float w = 0.0f;
    const char *end = len < 0 ? nullptr : text + len;
    for (const char *p = text; *p && (end == nullptr || p < end); ) {
        const unsigned cp = next_codepoint(&p);
        const std::int16_t gi = cp < 256 ? g_glyph_index[cp] : -1;
        if (gi >= 0) w += kFontGlyphs[gi].advance * kTextScale;
    }
    return (int)(w + 0.5f);
}

int r_text_height() {
    return (int)(kFontLineHeight * kTextScale + 0.5f);
}
