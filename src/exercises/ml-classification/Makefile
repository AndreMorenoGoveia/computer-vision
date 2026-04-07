CXX := g++
CXXFLAGS := -std=gnu++14 -fmax-errors=2 -O3 -s
CEKEIKON_OPENCV_ROOT := /home/andre/cekeikon5/opencv2cpu
CEKEIKON_ROOT := /home/andre/cekeikon5/cekeikon
OPENCV_FLAGS := $(shell pkg-config $(CEKEIKON_OPENCV_ROOT)/lib/pkgconfig/opencv.pc --libs --cflags)
CEKEIKON_FLAGS := -L$(CEKEIKON_OPENCV_ROOT)/lib -I$(CEKEIKON_OPENCV_ROOT)/include -lcekeikon2 -L$(CEKEIKON_ROOT)/bin -I$(CEKEIKON_ROOT)/src
BIN_DIR := bin
CPP_FILES := $(notdir $(wildcard *.cpp))

.PHONY: help clean force $(CPP_FILES)

help:
	@echo "Uso:"
	@echo "  make <arquivo.cpp>   Compila em $(BIN_DIR)/ e executa"
	@echo "  make clean           Remove os binarios gerados"
	@echo ""
	@echo "Arquivos disponiveis:"
	@printf "  %s\n" $(CPP_FILES)

clean:
	rm -rf $(BIN_DIR)

force:

define CPP_TEMPLATE
$(1): force
	mkdir -p $(BIN_DIR) results
	$(CXX) $(CXXFLAGS) $(1) -o $(BIN_DIR)/$(basename $(1)) $(CEKEIKON_FLAGS) $(OPENCV_FLAGS)
	./$(BIN_DIR)/$(basename $(1))

endef

$(foreach file,$(CPP_FILES),$(eval $(call CPP_TEMPLATE,$(file))))
