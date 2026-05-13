// MetalBackendCUDA.cpp
// Windows CUDA backend equivalent for MetalBackend.swift

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// ============================================================
// CUDA CHECK
// ============================================================

#define CUDA_CHECK(x)                                                     \
do {                                                                      \
    cudaError_t err = (x);                                                \
    if (err != cudaSuccess) {                                             \
        throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                     \
} while (0)

// ============================================================
// CUDA KERNEL
// ============================================================

__global__ void matmul_kernel(
    const float* weights,
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t cols
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // IMPORTANT:
    // CUDA launches rounded-up thread counts.
    // Prevent out-of-bounds access.
    if (gid >= rows) {
        return;
    }

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

    size_t weightsCapacityBytes = 0;
    size_t inputCapacityBytes = 0;
    size_t outputCapacityBytes = 0;

    int bufferRows = 0;
    int bufferCols = 0;

    bool weightsLoaded = false;

    ~CudaBackend() {
        destroyBuffers();
    }

    void ensureBuffers(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw std::runtime_error("Invalid rows/cols");
        }

        const size_t weightsSize = static_cast<size_t>(rows) *
                                   static_cast<size_t>(cols) *
                                   sizeof(float);

        const size_t inputSize = static_cast<size_t>(cols) *
                                 sizeof(float);

        const size_t outputSize = static_cast<size_t>(rows) *
                                  sizeof(float);

        bool ok =
            weightsBuffer != nullptr &&
            inputBuffer != nullptr &&
            outputBuffer != nullptr &&
            weightsCapacityBytes >= weightsSize &&
            inputCapacityBytes >= inputSize &&
            outputCapacityBytes >= outputSize;

        if (ok) {
            return;
        }

        // ====================================================
        // WEIGHTS
        // ====================================================

        if (weightsCapacityBytes < weightsSize) {

            if (weightsBuffer != nullptr) {
                CUDA_CHECK(cudaFree(weightsBuffer));
            }

            CUDA_CHECK(cudaMalloc(&weightsBuffer, weightsSize));

            weightsCapacityBytes = weightsSize;
        }

        // ====================================================
        // INPUT
        // ====================================================

        if (inputCapacityBytes < inputSize) {

            if (inputBuffer != nullptr) {
                CUDA_CHECK(cudaFree(inputBuffer));
            }

            CUDA_CHECK(cudaMalloc(&inputBuffer, inputSize));

            inputCapacityBytes = inputSize;
        }

        // ====================================================
        // OUTPUT
        // ====================================================

        if (outputCapacityBytes < outputSize) {

            if (outputBuffer != nullptr) {
                CUDA_CHECK(cudaFree(outputBuffer));
            }

            CUDA_CHECK(cudaMalloc(&outputBuffer, outputSize));

            outputCapacityBytes = outputSize;
        }

        bufferRows = rows;
        bufferCols = cols;
    }

    void destroyBuffers() {

        if (weightsBuffer != nullptr) {
            CUDA_CHECK(cudaFree(weightsBuffer));
            weightsBuffer = nullptr;
        }

        if (inputBuffer != nullptr) {
            CUDA_CHECK(cudaFree(inputBuffer));
            inputBuffer = nullptr;
        }

        if (outputBuffer != nullptr) {
            CUDA_CHECK(cudaFree(outputBuffer));
            outputBuffer = nullptr;
        }

        weightsCapacityBytes = 0;
        inputCapacityBytes = 0;
        outputCapacityBytes = 0;

        bufferRows = 0;
        bufferCols = 0;

        weightsLoaded = false;
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

        const size_t size =
            static_cast<size_t>(rows) *
            static_cast<size_t>(cols) *
            sizeof(float);

        CUDA_CHECK(cudaMemcpy(
            weightsBuffer,
            weights,
            size,
            cudaMemcpyHostToDevice
        ));

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

        CUDA_CHECK(cudaMemcpy(
            dst + dstStart,
            weightsBuffer + srcStart,
            static_cast<size_t>(length) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
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

        CUDA_CHECK(cudaMemcpy(
            dst + dstStart,
            outputBuffer + srcStart,
            static_cast<size_t>(length) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }

    // ============================================================
    // INTERNAL KERNEL LAUNCH
    // ============================================================

    void launchMatmul(
        float* input,
        float* output,
        int rows,
        int cols
    ) {
        constexpr int THREADS = 256;

        const int blocks =
            (rows + THREADS - 1) / THREADS;

        matmul_kernel<<<blocks, THREADS>>>(
            weightsBuffer,
            input,
            output,
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(cols)
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
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

        ensureBuffers(rows, cols);

        CUDA_CHECK(cudaMemcpy(
            inputBuffer,
            input,
            static_cast<size_t>(cols) * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        launchMatmul(
            inputBuffer,
            outputBuffer,
            rows,
            cols
        );

        CUDA_CHECK(cudaMemcpy(
            output,
            outputBuffer,
            static_cast<size_t>(rows) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
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
        assert(rows > 0);
        assert(cols > 0);

        ensureBuffers(rows, cols);

        launchMatmul(
            input,
            outputBuffer,
            rows,
            cols
        );

        CUDA_CHECK(cudaMemcpy(
            output,
            outputBuffer,
            static_cast<size_t>(rows) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
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
        assert(rows > 0);
        assert(cols > 0);

        ensureBuffers(rows, cols);

        CUDA_CHECK(cudaMemcpy(
            inputBuffer,
            input,
            static_cast<size_t>(cols) * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        launchMatmul(
            inputBuffer,
            output,
            rows,
            cols
        );
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
        assert(rows > 0);
        assert(cols > 0);

        ensureBuffers(rows, cols);

        launchMatmul(
            input,
            output,
            rows,
            cols
        );
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

    CUDA_CHECK(cudaMalloc(
        &ptr,
        static_cast<size_t>(size) * sizeof(float)
    ));

    return ptr;
}

__declspec(dllexport)
void cuda_destroy_buffer(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

__declspec(dllexport)
void cuda_set_range(
    void* ptr,
    const float* src,
    int dstStart,
    int srcStart,
    int count
) {
    CUDA_CHECK(cudaMemcpy(
        static_cast<float*>(ptr) + dstStart,
        src + srcStart,
        static_cast<size_t>(count) * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

__declspec(dllexport)
void cuda_copy_data_to(
    void* ptr,
    float* dst,
    int srcStart,
    int dstStart,
    int count
) {
    CUDA_CHECK(cudaMemcpy(
        dst + dstStart,
        static_cast<float*>(ptr) + srcStart,
        static_cast<size_t>(count) * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

__declspec(dllexport)
void add_gpu_buffer_to_cpu_buffer(
    void* gpuBuffer,
    float* cpuBuffer,
    int length
) {
    float* tmp = new float[length];

    CUDA_CHECK(cudaMemcpy(
        tmp,
        gpuBuffer,
        static_cast<size_t>(length) * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < length; i++) {
        cpuBuffer[i] += tmp[i];
    }

    delete[] tmp;
}

}