EMCC   ?= emcc
EMXX   ?= em++
MICROUI = third_party/microui
BUILD  ?= build
PORT   ?= 8000

OBJECTS = $(BUILD)/main.o $(BUILD)/renderer.o $(BUILD)/cells.o $(BUILD)/microui.o

INCLUDES = -I$(MICROUI)/src -I$(MICROUI)/demo -Isrc

COMMON = $(INCLUDES) -Os -msimd128 --use-port=emdawnwebgpu

CXXFLAGS = -std=c++17 $(COMMON)

# microui is upstream C and stays that way; only our own sources are C++.
CFLAGS = -std=c99 $(COMMON)

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

$(BUILD)/main.o: src/main.cpp src/renderer.h src/cells.h $(MICROUI)/src/microui.c | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/renderer.o: src/renderer.cpp src/renderer.h $(MICROUI)/src/microui.c | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/cells.o: src/cells.cpp src/cells.h | $(BUILD)
	$(EMXX) $(CXXFLAGS) -c $< -o $@

# Depends on microui.c only -- the clone produces atlas.inl at the same time.
$(BUILD)/microui.o: $(MICROUI)/src/microui.c | $(BUILD)
	$(EMCC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

serve: all
	@echo "http://localhost:$(PORT)/"
	python3 -m http.server $(PORT) --directory $(BUILD)

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf third_party
