// MetalBackendCUDA.cpp
// Windows CUDA backend equivalent for MetalBackend.swift

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// ============================================================
// CUDA KERNEL
// ============================================================

__global__ void matmul_kernel(
    const float* weights,
    const float* input,
    float* output,
    uint32_t cols
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    uint32_t rowOffset = gid * cols;

    for (uint32_t i = 0; i < cols; i++) {
        sum += weights[rowOffset + i] * input[i];
    }

    output[gid] = sum;
}

// ============================================================
// CUDA BACKEND
// ============================================================

class CudaBackend {
public:

    float* weightsBuffer = nullptr;
    float* inputBuffer = nullptr;
    float* outputBuffer = nullptr;

    int bufferRows = 0;
    int bufferCols = 0;

    bool weightsLoaded = false;

    ~CudaBackend() {
        destroyBuffers();
    }

    void ensureBuffers(int rows, int cols) {
        if (bufferRows >= rows &&
            bufferCols >= cols &&
            weightsBuffer != nullptr &&
            inputBuffer != nullptr &&
            outputBuffer != nullptr) {
            return;
        }

        const size_t weightsSize = rows * cols * sizeof(float);
        const size_t inputSize = cols * sizeof(float);
        const size_t outputSize = rows * sizeof(float);

        if (weightsBuffer == nullptr ||
            (bufferRows * bufferCols * sizeof(float)) < weightsSize) {

            if (weightsBuffer != nullptr) {
                cudaFree(weightsBuffer);
            }

            cudaMalloc(&weightsBuffer, weightsSize);
        }

        if (inputBuffer == nullptr ||
            (bufferCols * sizeof(float)) < inputSize) {

            if (inputBuffer != nullptr) {
                cudaFree(inputBuffer);
            }

            cudaMalloc(&inputBuffer, inputSize);
        }

        if (outputBuffer == nullptr ||
            (bufferRows * sizeof(float)) < outputSize) {

            if (outputBuffer != nullptr) {
                cudaFree(outputBuffer);
            }

            cudaMalloc(&outputBuffer, outputSize);
        }

        bufferRows = (rows > bufferRows) ? rows : bufferRows;
        bufferCols = (cols > bufferCols) ? cols : bufferCols;
    }

    void destroyBuffers() {
        if (weightsBuffer != nullptr) {
            cudaFree(weightsBuffer);
            weightsBuffer = nullptr;
        }

        if (inputBuffer != nullptr) {
            cudaFree(inputBuffer);
            inputBuffer = nullptr;
        }

        if (outputBuffer != nullptr) {
            cudaFree(outputBuffer);
            outputBuffer = nullptr;
        }
    }

    // ============================================================
    // WEIGHTS
    // ============================================================

    void setWeights(
        const float* weights,
        int rows,
        int cols
    ) {
        ensureBuffers(rows, cols);

        const size_t size = rows * cols * sizeof(float);

        cudaMemcpy(
            weightsBuffer,
            weights,
            size,
            cudaMemcpyHostToDevice
        );

        weightsLoaded = true;
    }

    // ============================================================
    // COPY WEIGHTS
    // ============================================================

    void copyWeights(
        float* dst,
        int dstStart,
        int srcStart,
        int length
    ) {
        assert(length >= 0);
        assert(dstStart >= 0);
        assert(srcStart >= 0);

        const int totalWeights = bufferRows * bufferCols;

        assert(srcStart + length <= totalWeights);

        cudaMemcpy(
            dst + dstStart,
            weightsBuffer + srcStart,
            length * sizeof(float),
            cudaMemcpyDeviceToHost
        );
    }

    // ============================================================
    // COPY OUTPUT
    // ============================================================

    void copyOutput(
        float* dst,
        int dstStart,
        int srcStart,
        int length
    ) {
        assert(length >= 0);
        assert(dstStart >= 0);
        assert(srcStart >= 0);

        assert(srcStart + length <= bufferRows);

        cudaMemcpy(
            dst + dstStart,
            outputBuffer + srcStart,
            length * sizeof(float),
            cudaMemcpyDeviceToHost
        );
    }

    // ============================================================
    // MATMUL
    // ============================================================

    void matmul(
        const float* input,
        float* output,
        int rows,
        int cols
    ) {
        assert(weightsLoaded);
        assert(rows > 0);
        assert(cols > 0);

        cudaMemcpy(
            inputBuffer,
            input,
            cols * sizeof(float),
            cudaMemcpyHostToDevice
        );

        constexpr int THREADS = 256;

        int blocks = (rows + THREADS - 1) / THREADS;

        matmul_kernel<<<blocks, THREADS>>>(
            weightsBuffer,
            inputBuffer,
            outputBuffer,
            static_cast<uint32_t>(cols)
        );

        cudaDeviceSynchronize();

        cudaMemcpy(
            output,
            outputBuffer,
            rows * sizeof(float),
            cudaMemcpyDeviceToHost
        );
    }

    // ============================================================
    // GPU INPUT BUFFER
    // ============================================================

    void matmulInputCudaBuffer(
        float* input,
        float* output,
        int rows,
        int cols
    ) {
        assert(weightsLoaded);

        constexpr int THREADS = 256;

        int blocks = (rows + THREADS - 1) / THREADS;

        matmul_kernel<<<blocks, THREADS>>>(
            weightsBuffer,
            input,
            outputBuffer,
            static_cast<uint32_t>(cols)
        );

        cudaDeviceSynchronize();

        cudaMemcpy(
            output,
            outputBuffer,
            rows * sizeof(float),
            cudaMemcpyDeviceToHost
        );
    }

    // ============================================================
    // GPU OUTPUT BUFFER
    // ============================================================

    void matmulOutputCudaBuffer(
        const float* input,
        float* output,
        int rows,
        int cols
    ) {
        assert(weightsLoaded);

        cudaMemcpy(
            inputBuffer,
            input,
            cols * sizeof(float),
            cudaMemcpyHostToDevice
        );

        constexpr int THREADS = 256;

        int blocks = (rows + THREADS - 1) / THREADS;

        matmul_kernel<<<blocks, THREADS>>>(
            weightsBuffer,
            inputBuffer,
            output,
            static_cast<uint32_t>(cols)
        );

        cudaDeviceSynchronize();
    }

    // ============================================================
    // GPU INPUT + OUTPUT BUFFERS
    // ============================================================

    void matmulInputOutputCudaBuffer(
        float* input,
        float* output,
        int rows,
        int cols
    ) {
        assert(weightsLoaded);

        constexpr int THREADS = 256;

        int blocks = (rows + THREADS - 1) / THREADS;

        matmul_kernel<<<blocks, THREADS>>>(
            weightsBuffer,
            input,
            output,
            static_cast<uint32_t>(cols)
        );

        cudaDeviceSynchronize();
    }
};

// ============================================================
// C API
// ============================================================

extern "C" {

__declspec(dllexport)
void* cuda_create() {
    return new CudaBackend();
}

__declspec(dllexport)
void cuda_destroy(void* ptr) {
    delete static_cast<CudaBackend*>(ptr);
}

__declspec(dllexport)
void cuda_set_weights(
    void* ptr,
    const float* weights,
    int32_t rows,
    int32_t cols
) {
    auto* backend = static_cast<CudaBackend*>(ptr);

    backend->setWeights(
        weights,
        static_cast<int>(rows),
        static_cast<int>(cols)
    );
}

__declspec(dllexport)
void cuda_copy_weights(
    void* ptr,
    float* dst,
    int dstStart,
    int srcStart,
    int length
) {
    auto* backend = static_cast<CudaBackend*>(ptr);

    backend->copyWeights(
        dst,
        dstStart,
        srcStart,
        length
    );
}

__declspec(dllexport)
void cuda_copy_output(
    void* ptr,
    float* dst,
    int dstStart,
    int srcStart,
    int length
) {
    auto* backend = static_cast<CudaBackend*>(ptr);

    backend->copyOutput(
        dst,
        dstStart,
        srcStart,
        length
    );
}

__declspec(dllexport)
void cuda_matmul(
    void* ptr,
    const float* input,
    float* output,
    int rows,
    int cols
) {
    auto* backend = static_cast<CudaBackend*>(ptr);

    backend->matmul(
        input,
        output,
        rows,
        cols
    );
}

__declspec(dllexport)
void cuda_matmul_input_cudabuffer(
    void* ptr,
    void* inputBuffer,
    float* output,
    int rows,
    int cols
) {
    auto* backend = static_cast<CudaBackend*>(ptr);

    backend->matmulInputCudaBuffer(
        static_cast<float*>(inputBuffer),
        output,
        rows,
        cols
    );
}

__declspec(dllexport)
void cuda_matmul_output_cudabuffer(
    void* ptr,
    const float* input,
    void* outputBuffer,
    int rows,
    int cols
) {
    auto* backend = static_cast<CudaBackend*>(ptr);

    backend->matmulOutputCudaBuffer(
        input,
        static_cast<float*>(outputBuffer),
        rows,
        cols
    );
}

__declspec(dllexport)
void cuda_matmul_input_output_cudabuffer(
    void* ptr,
    void* inputBuffer,
    void* outputBuffer,
    int rows,
    int cols
) {
    auto* backend = static_cast<CudaBackend*>(ptr);

    backend->matmulInputOutputCudaBuffer(
        static_cast<float*>(inputBuffer),
        static_cast<float*>(outputBuffer),
        rows,
        cols
    );
}

__declspec(dllexport)
void* cuda_create_float_buffer(int size) {
    float* ptr = nullptr;

    cudaMalloc(&ptr, size * sizeof(float));

    return ptr;
}

__declspec(dllexport)
void cuda_destroy_buffer(void* ptr) {
    cudaFree(ptr);
}

__declspec(dllexport)
void cuda_set_range(
    void* ptr,
    const float* src,
    int dstStart,
    int srcStart,
    int count
) {
    cudaMemcpy(
        static_cast<float*>(ptr) + dstStart,
        src + srcStart,
        count * sizeof(float),
        cudaMemcpyHostToDevice
    );
}

__declspec(dllexport)
void cuda_copy_data_to(
    void* ptr,
    float* dst,
    int srcStart,
    int dstStart,
    int count
) {
    cudaMemcpy(
        dst + dstStart,
        static_cast<float*>(ptr) + srcStart,
        count * sizeof(float),
        cudaMemcpyDeviceToHost
    );
}

__declspec(dllexport)
void add_gpu_buffer_to_cpu_buffer(
    void* gpuBuffer,
    float* cpuBuffer,
    int length
) {
    float* tmp = new float[length];

    cudaMemcpy(
        tmp,
        gpuBuffer,
        length * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < length; i++) {
        cpuBuffer[i] += tmp[i];
    }

    delete[] tmp;
}

}