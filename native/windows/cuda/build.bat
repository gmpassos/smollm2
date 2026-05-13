
rmdir /s /q build 2>nul

mkdir build

nvcc -O3 --shared -o build/cuda_backend.dll MetalBackendCUDA.cu
