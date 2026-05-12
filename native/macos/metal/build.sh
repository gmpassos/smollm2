#!/usr/bin/env bash

set -e
set -x

rm -rf build

mkdir -p build

swiftc \
-O \
-whole-module-optimization \
-emit-library \
MetalBackend.swift \
-o build/libsmollm2_metal.dylib \
-framework Foundation \
-framework Metal