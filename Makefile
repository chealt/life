EMCC   ?= emcc
EMXX   ?= em++
BUILD  ?= build
PORT   ?= 8000

OBJECTS = $(BUILD)/main.o $(BUILD)/renderer.o $(BUILD)/cells.o \
          $(BUILD)/vessel.o $(BUILD)/ui.o

INCLUDES = -Isrc

COMMON = $(INCLUDES) -Os -msimd128 --use-port=emdawnwebgpu

CXXFLAGS = -std=c++23 $(COMMON)

LDFLAGS = \
	--use-port=emdawnwebgpu \
	-sALLOW_MEMORY_GROWTH=1 \
	-sENVIRONMENT=web \
	-sFILESYSTEM=0 \
	-sEXIT_RUNTIME=0 \
	-Os --closure 0 \
	--shell-file src/shell.html

.PHONY: all serve clean

all: $(BUILD)/index.html

$(BUILD)/index.html: $(OBJECTS) src/shell.html
	$(EMXX) $(OBJECTS) $(LDFLAGS) -o $@

$(BUILD)/main.o: src/main.cpp src/renderer.h src/cells.h src/vessel.h src/ui.h \
                 src/camera.h src/math3d.h | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/renderer.o: src/renderer.cpp src/renderer.h src/font_atlas.h | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/cells.o: src/cells.cpp src/cells.h src/math3d.h | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/vessel.o: src/vessel.cpp src/vessel.h src/cells.h src/math3d.h | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/ui.o: src/ui.cpp src/ui.h src/renderer.h | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

serve: all
	@echo "http://localhost:$(PORT)/"
	python3 -m http.server $(PORT) --directory $(BUILD)

clean:
	rm -rf $(BUILD)
