#!/usr/bin/env bash

set -e
set -x

rm -rf build

mkdir -p build

swiftc \
-O \
-whole-module-optimization \
-emit-library \
CpuBackend.swift \
-o build/libsmollm2_cpu.dylib \
-framework Accelerate \
-framework Foundation