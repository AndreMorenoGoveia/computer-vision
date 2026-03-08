CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4)
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
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(1) -o $(BIN_DIR)/$(basename $(1)) $(OPENCV_FLAGS)
	./$(BIN_DIR)/$(basename $(1))

endef

$(foreach file,$(CPP_FILES),$(eval $(call CPP_TEMPLATE,$(file))))
