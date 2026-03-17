CXX := g++
CXXFLAGS := -std=gnu++14 -fmax-errors=2 -O3 -s
CEKEIKON_OPENCV_ROOT := /home/andre/cekeikon5/opencv2cpu
CEKEIKON_ROOT := /home/andre/cekeikon5/cekeikon
OPENCV_FLAGS := $(shell pkg-config $(CEKEIKON_OPENCV_ROOT)/lib/pkgconfig/opencv.pc --libs --cflags)
CEKEIKON_FLAGS := -L$(CEKEIKON_OPENCV_ROOT)/lib -I$(CEKEIKON_OPENCV_ROOT)/include -lcekeikon2 -L$(CEKEIKON_ROOT)/bin -I$(CEKEIKON_ROOT)/src
CPP_FILES := $(notdir $(wildcard *.cpp))

.PHONY: help clean force $(CPP_FILES)

help:
	@echo "Uso:"
	@echo "  make <arquivo.cpp>   Compila na raiz e executa"
	@echo "  make clean           Remove os binarios gerados"
	@echo ""
	@echo "Arquivos disponiveis:"
	@printf "  %s\n" $(CPP_FILES)

clean:
	rm -f $(basename $(CPP_FILES))

force:

define CPP_TEMPLATE
$(1): force
	mkdir -p results
	$(CXX) $(CXXFLAGS) $(1) -o $(basename $(1)) $(CEKEIKON_FLAGS) $(OPENCV_FLAGS)
	./$(basename $(1))

endef

$(foreach file,$(CPP_FILES),$(eval $(call CPP_TEMPLATE,$(file))))
