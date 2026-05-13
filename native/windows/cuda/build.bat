
mkdir build
nvcc -O3 --shared -o build/cuda_backend.dll MetalBackendCUDA.cu
