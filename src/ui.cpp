#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "ui.h"

namespace {

// Palette. One accent, a near-black panel, and enough separation between the
// resting and hovered states to be readable without an outline.
constexpr UiColor kPanel      { 16, 17, 22, 235 };
constexpr UiColor kTitleBar   { 24, 26, 33, 255 };
constexpr UiColor kTitleText  { 238, 240, 245, 255 };
constexpr UiColor kText       { 196, 200, 210, 255 };
constexpr UiColor kMuted      { 128, 133, 147, 255 };
constexpr UiColor kTrack      { 30, 32, 40, 255 };
constexpr UiColor kFill       { 168, 38, 44, 255 };
constexpr UiColor kFillHot    { 206, 52, 58, 255 };
constexpr UiColor kKnob       { 236, 238, 243, 255 };
constexpr UiColor kButton     { 38, 41, 51, 255 };
constexpr UiColor kButtonHot  { 52, 56, 70, 255 };
constexpr UiColor kButtonDown { 200, 45, 50, 255 };
constexpr UiColor kRule       { 44, 48, 60, 255 };

constexpr int kPad      = 12;
constexpr int kRowH     = 26;
constexpr int kGap      = 6;
constexpr int kTitleH   = 34;
constexpr int kLabelW   = 104;
constexpr int kKnobW    = 10;
// The value is right-aligned in a reserved column. Without this the number
// grows leftwards over the track as it gets wider.
constexpr int kValueW   = 58;
constexpr int kValueGap = 8;

struct State {
    UiInput input;
    int     screen_w = 0, screen_h = 0;

    // Panel being built.
    int panel_x = 0, panel_y = 0, panel_w = 0;
    int cursor_y = 0;

    // Which control owns the drag. Widgets are identified by their label
    // pointer, which is a string literal and so stable across frames.
    const void *active = nullptr;

    // A panel's height is only known once its last widget has been laid out,
    // but the background has to be drawn first to sit behind them. Carry the
    // measurement over from the previous frame; the panel is the same shape
    // frame to frame, so the only wrong one is the very first.
    int  panel_h = 220;
    UiRect last_panel{};
};

State g;

bool hit(int x, int y, int w, int h) {
    return g.input.mouse_x >= x && g.input.mouse_x < x + w &&
           g.input.mouse_y >= y && g.input.mouse_y < y + h;
}

// Reserve the next row and return its rectangle.
UiRect row(int height) {
    UiRect r{ g.panel_x + kPad, g.cursor_y, g.panel_w - kPad * 2, height };
    g.cursor_y += height + kGap;
    return r;
}

void text_at(const char *s, int x, int y, int h, UiColor c) {
    // Centre vertically on the row.
    r_draw_text(s, x, y + (h - r_text_height()) / 2, c);
}

bool slider(const char *label, float *value, float lo, float hi,
            bool integral, const char *value_text) {
    const UiRect r = row(kRowH);
    const int track_x = r.x + kLabelW;
    const int track_w = std::max(24, r.w - kLabelW - kValueW - kValueGap);

    const bool over = hit(track_x, r.y, track_w, r.h);
    if (over && g.input.mouse_pressed) g.active = label;
    if (!g.input.mouse_down && g.active == label) g.active = nullptr;

    bool changed = false;
    if (g.active == label) {
        const float t = std::clamp(
            static_cast<float>(g.input.mouse_x - track_x) / static_cast<float>(track_w),
            0.0f, 1.0f);
        float next = lo + t * (hi - lo);
        if (integral) next = std::round(next);
        if (next != *value) { *value = next; changed = true; }
    }

    const float span = hi - lo;
    const float t = span > 0.0f ? std::clamp((*value - lo) / span, 0.0f, 1.0f) : 0.0f;
    const int fill_w = static_cast<int>(t * static_cast<float>(track_w - kKnobW));

    // Track, filled portion, then the knob.
    const int bar_y = r.y + r.h / 2 - 3;
    r_draw_rect(UiRect{ track_x, bar_y, track_w, 6 }, kTrack);
    r_draw_rect(UiRect{ track_x, bar_y, fill_w + kKnobW / 2, 6 },
                (over || g.active == label) ? kFillHot : kFill);
    r_draw_rect(UiRect{ track_x + fill_w, r.y + 3, kKnobW, r.h - 6 }, kKnob);

    text_at(label, r.x, r.y, r.h, kText);

    char buf[32];
    if (value_text)    std::snprintf(buf, sizeof(buf), "%s", value_text);
    else if (integral) std::snprintf(buf, sizeof(buf), "%d", static_cast<int>(*value));
    else               std::snprintf(buf, sizeof(buf), "%.2f", *value);
    const int tw = r_text_width(buf);
    text_at(buf, r.x + r.w - tw, r.y, r.h, kMuted);

    return changed;
}

}  // namespace

void ui_begin(const UiInput &input, int width, int height) {
    g.input = input;
    g.screen_w = width;
    g.screen_h = height;
}

void ui_end() {
    if (!g.input.mouse_down) g.active = nullptr;
}

// Tested against live pointer coordinates rather than the ones captured at the
// start of the frame: a press can arrive between frames, and routing it to the
// camera when it landed on a slider would be felt immediately.
bool ui_captures_mouse(int x, int y) {
    if (g.active != nullptr) return true;
    const UiRect &p = g.last_panel;
    return x >= p.x && x < p.x + p.w && y >= p.y && y < p.y + p.h;
}

void ui_panel_begin(const char *title, int x, int y, int w) {
    g.panel_x = x;
    g.panel_y = y;
    g.panel_w = w;
    g.cursor_y = y + kTitleH + kPad;

    // Widgets are submitted after this and the renderer preserves submission
    // order, so the background lands behind them.
    r_draw_rect(UiRect{ x, y, w, g.panel_h }, kPanel);
    r_draw_rect(UiRect{ x, y, w, kTitleH }, kTitleBar);
    text_at(title, x + kPad, y, kTitleH, kTitleText);
}

void ui_panel_end() {
    g.panel_h = g.cursor_y - g.panel_y + kPad - kGap;
    g.last_panel = UiRect{ g.panel_x, g.panel_y, g.panel_w, g.panel_h };
}

void ui_category(const char *text) {
    const UiRect r = row(22);
    text_at(text, r.x, r.y, r.h, kMuted);
    r_draw_rect(UiRect{ r.x, r.y + r.h - 1, r.w, 1 }, kRule);
}

void ui_label(const char *text) {
    const UiRect r = row(kRowH);
    text_at(text, r.x, r.y, r.h, kText);
}

bool ui_slider_int(const char *label, int *value, int lo, int hi,
                   const char *value_text) {
    float v = static_cast<float>(*value);
    const bool changed = slider(label, &v, static_cast<float>(lo),
                                static_cast<float>(hi), true, value_text);
    *value = static_cast<int>(v);
    return changed;
}

bool ui_slider_float(const char *label, float *value, float lo, float hi) {
    return slider(label, value, lo, hi, false, nullptr);
}

bool ui_button(const char *label) {
    const UiRect r = row(kRowH + 4);
    const bool over = hit(r.x, r.y, r.w, r.h);
    const bool down = over && g.input.mouse_down;

    r_draw_rect(r, down ? kButtonDown : (over ? kButtonHot : kButton));
    const int tw = r_text_width(label);
    text_at(label, r.x + (r.w - tw) / 2, r.y, r.h, kTitleText);

    return over && g.input.mouse_pressed;
}
