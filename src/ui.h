// A small immediate-mode UI, drawn through renderer.h.
//
// Widgets are called every frame and hold no state of their own; the only
// state kept here is which control the pointer is currently interacting with.
#ifndef UI_H
#define UI_H

#include "renderer.h"

struct UiInput {
    int  mouse_x = 0, mouse_y = 0;  // logical units
    bool mouse_down = false;        // held this frame
    bool mouse_pressed = false;     // went down this frame
};

void ui_begin(const UiInput &input, int width, int height);
void ui_end();

// True when the pointer at (x, y) is over a panel, or a control is mid-drag,
// so the camera knows to leave the event alone.
bool ui_captures_mouse(int x, int y);

void ui_panel_begin(const char *title, int x, int y, int w);
void ui_panel_end();

void ui_category(const char *text);
void ui_label(const char *text);
// `value_text` overrides the number shown on the right, for units and the
// like; pass nullptr for the plain integer.
bool ui_slider_int(const char *label, int *value, int lo, int hi,
                   const char *value_text = nullptr);
bool ui_slider_float(const char *label, float *value, float lo, float hi);
// Logarithmic, for ranges spanning orders of magnitude where a linear track
// would leave everything but the top end unreachable.
bool ui_slider_log(const char *label, double *value, double lo, double hi,
                   const char *value_text = nullptr);
bool ui_button(const char *label);

#endif
