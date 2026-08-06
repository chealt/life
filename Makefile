EMCC   ?= emcc
# emdawnwebgpu implements the webgpu.h C API in C++, so the *link* needs em++.
# Compilation stays C99 -- only the final link pulls the C++ runtime in.
EMXX   ?= em++
MICROUI = third_party/microui
BUILD  ?= build
PORT   ?= 8000

SOURCES = src/main.c src/renderer.c src/cells.c $(MICROUI)/src/microui.c
OBJECTS = $(BUILD)/main.o $(BUILD)/renderer.o $(BUILD)/cells.o $(BUILD)/microui.o

CFLAGS = \
	-std=c99 \
	-I$(MICROUI)/src -I$(MICROUI)/demo -Isrc \
	-Os -msimd128 \
	--use-port=emdawnwebgpu

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

$(BUILD)/main.o: src/main.c src/renderer.h src/cells.h $(MICROUI)/src/microui.c | $(BUILD)
	$(EMCC) $(CFLAGS) -c $< -o $@

$(BUILD)/cells.o: src/cells.c src/cells.h | $(BUILD)
	$(EMCC) $(CFLAGS) -c $< -o $@

$(BUILD)/renderer.o: src/renderer.c src/renderer.h $(MICROUI)/src/microui.c | $(BUILD)
	$(EMCC) $(CFLAGS) -c $< -o $@

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
