CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
OPENCV   := $(shell pkg-config --cflags opencv4) -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

HEADERS  := dct_denoise.hpp wavelet_denoise.hpp toy_jets.hpp

all: denoise_test wavelet_test gw_test

%: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

test: all
	./denoise_test
	./wavelet_test
	./gw_test

clean:
	rm -f denoise_test wavelet_test gw_test denoise_test.png wavelet_test.png gw_test.png

.PHONY: all test clean
