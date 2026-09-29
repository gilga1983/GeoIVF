CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Werror -fPIC
URING ?= 0
ifeq ($(URING),1)
CPPFLAGS += -DGEOIVF_URING
LDLIBS += -luring
endif
.PHONY: all test clean
all: build/libgeoivf_io.so
build/libgeoivf_io.so: native/reader.cpp Makefile
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -shared $< -o $@ $(LDLIBS)
test: all
	python -m pytest -q
clean:
	rm -f build/libgeoivf_io.so
