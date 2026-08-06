EMCC   ?= emcc
EMXX   ?= em++
MICROUI = third_party/microui
BUILD  ?= build
PORT   ?= 8000

OBJECTS = $(BUILD)/main.o $(BUILD)/renderer.o $(BUILD)/cells.o \
          $(BUILD)/field.o $(BUILD)/ui.o

INCLUDES = -I$(MICROUI)/src -I$(MICROUI)/demo -Isrc

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

.PHONY: all deps serve clean distclean

all: $(BUILD)/index.html

deps: $(MICROUI)/src/microui.c

$(MICROUI)/src/microui.c:
	git clone --depth 1 https://github.com/rxi/microui $(MICROUI)

$(BUILD)/index.html: $(OBJECTS) src/shell.html
	$(EMXX) $(OBJECTS) $(LDFLAGS) -o $@

$(BUILD)/main.o: src/main.cpp src/renderer.h src/cells.h src/field.h src/ui.h \
                 src/camera.h src/math3d.h $(MICROUI)/src/microui.c | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

# Depends on microui.c only -- the clone produces atlas.inl at the same time,
# and the atlas is the sole remaining use of that dependency.
$(BUILD)/renderer.o: src/renderer.cpp src/renderer.h $(MICROUI)/src/microui.c | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/cells.o: src/cells.cpp src/cells.h src/math3d.h | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/field.o: src/field.cpp src/field.h src/cells.h src/math3d.h | $(BUILD)
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

distclean: clean
	rm -rf third_party
