// MetalBackendCUDA.cu
// Windows CUDA backend equivalent for MetalBackend.swift

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// ============================================================
// CUDA CHECK HELPERS
// ============================================================

#define CUDA_CHECK_THROW(x)                                      \
do {                                                             \
    cudaError_t err = (x);                                       \
    if (err != cudaSuccess) {                                    \
        throw std::runtime_error(cudaGetErrorString(err));       \
    }                                                            \
} while (0)

#define CUDA_CHECK_RETURN(x)                                     \
do {                                                             \
    cudaError_t err = (x);                                       \
    if (err != cudaSuccess) {                                    \
        return;                                                  \
    }                                                            \
} while (0)

#define CUDA_CHECK_RETURN_NULL(x)                                \
do {                                                             \
    cudaError_t err = (x);                                       \
    if (err != cudaSuccess) {                                    \
        return nullptr;                                          \
    }                                                            \
} while (0)
// ============================================================
// SMID HELPERS
// ============================================================

#if defined(__CUDA_ARCH__)

__device__ __forceinline__
uint32_t cuda_get_smid() {

    uint32_t smid;

    asm volatile(
        "mov.u32 %0, %smid;"
        : "=r"(smid)
    );

    return smid;
}

#else

static inline
uint32_t cuda_get_smid() {
    return 0;
}

#endif

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
    uint32_t gid =
        blockIdx.x * blockDim.x + threadIdx.x;

    // IMPORTANT:
    // CUDA launches rounded-up thread counts.
    // Prevent out-of-bounds access.
    if (gid >= rows) {
        return;
    }

    // ========================================================
    // SMID COMPUTATION
    // ========================================================

    const uint32_t smid =
        cuda_get_smid();

    // Prevent compiler from optimizing away SMID fetch.
    // This keeps SMID computation active with near-zero cost.
    if (smid == 0xFFFFFFFFu) {
        return;
    }

    float sum = 0.0f;

    const uint32_t rowOffset =
        gid * cols;

    // ========================================================
    // FLOAT4 SECTION
    // ========================================================

    const uint32_t vecCols =
        cols / 4;

    const float4* weights4 =
        reinterpret_cast<const float4*>(
            weights + rowOffset
        );

    const float4* input4 =
        reinterpret_cast<const float4*>(
            input
        );

    for (uint32_t i = 0; i < vecCols; i++) {

        float4 w = weights4[i];
        float4 v = input4[i];

        sum += w.x * v.x;
        sum += w.y * v.y;
        sum += w.z * v.z;
        sum += w.w * v.w;
    }

    // ========================================================
    // TAIL SECTION
    // ========================================================

    const uint32_t tailStart =
        vecCols * 4;

    for (
        uint32_t i = tailStart;
        i < cols;
        i++
    ) {
        sum +=
            weights[rowOffset + i] *
            input[i];
    }

    output[gid] = sum;
}

// ============================================================
// COMPUTE LOGITS KERNEL
// ============================================================

__global__ void compute_logits_kernel(
    const float* x,
    const float* embed,
    float* logits,
    uint32_t vocabSize,
    uint32_t hiddenSize
) {
    uint32_t gid =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= vocabSize) {
        return;
    }

    // ========================================================
    // SMID COMPUTATION
    // ========================================================

    const uint32_t smid =
        cuda_get_smid();

    if (smid == 0xFFFFFFFFu) {
        return;
    }

    float sum = 0.0f;

    const uint32_t rowOffset =
        gid * hiddenSize;

    // ========================================================
    // FLOAT4 SECTION
    // ========================================================

    const uint32_t vecCols =
        hiddenSize / 4;

    const float4* x4 =
        reinterpret_cast<const float4*>(
            x
        );

    const float4* embed4 =
        reinterpret_cast<const float4*>(
            embed + rowOffset
        );

    for (uint32_t i = 0; i < vecCols; i++) {

        float4 xv = x4[i];
        float4 ev = embed4[i];

        sum += xv.x * ev.x;
        sum += xv.y * ev.y;
        sum += xv.z * ev.z;
        sum += xv.w * ev.w;
    }

    // ========================================================
    // TAIL SECTION
    // ========================================================

    const uint32_t tailStart =
        vecCols * 4;

    for (
        uint32_t i = tailStart;
        i < hiddenSize;
        i++
    ) {
        sum +=
            x[i] *
            embed[rowOffset + i];
    }

    logits[gid] = sum;
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
            throw std::runtime_error(
                "Invalid rows/cols"
            );
        }

        const size_t weightsSize =
            static_cast<size_t>(rows) *
            static_cast<size_t>(cols) *
            sizeof(float);

        const size_t inputSize =
            static_cast<size_t>(cols) *
            sizeof(float);

        const size_t outputSize =
            static_cast<size_t>(rows) *
            sizeof(float);

        bool alreadyValid =
            weightsBuffer != nullptr &&
            inputBuffer != nullptr &&
            outputBuffer != nullptr &&
            weightsCapacityBytes >= weightsSize &&
            inputCapacityBytes >= inputSize &&
            outputCapacityBytes >= outputSize;

        if (alreadyValid) {
            return;
        }

        // ====================================================
        // WEIGHTS
        // ====================================================

        if (weightsCapacityBytes < weightsSize) {

            if (weightsBuffer != nullptr) {
                CUDA_CHECK_THROW(
                    cudaFree(weightsBuffer)
                );
            }

            CUDA_CHECK_THROW(cudaMalloc(
                reinterpret_cast<void**>(&weightsBuffer),
                weightsSize
            ));

            weightsCapacityBytes = weightsSize;
        }

        // ====================================================
        // INPUT
        // ====================================================

        if (inputCapacityBytes < inputSize) {

            if (inputBuffer != nullptr) {
                CUDA_CHECK_THROW(
                    cudaFree(inputBuffer)
                );
            }

            CUDA_CHECK_THROW(cudaMalloc(
                reinterpret_cast<void**>(&inputBuffer),
                inputSize
            ));

            inputCapacityBytes = inputSize;
        }

        // ====================================================
        // OUTPUT
        // ====================================================

        if (outputCapacityBytes < outputSize) {

            if (outputBuffer != nullptr) {
                CUDA_CHECK_THROW(
                    cudaFree(outputBuffer)
                );
            }

            CUDA_CHECK_THROW(cudaMalloc(
                reinterpret_cast<void**>(&outputBuffer),
                outputSize
            ));

            outputCapacityBytes = outputSize;
        }

        bufferRows = rows;
        bufferCols = cols;
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

        CUDA_CHECK_THROW(cudaMemcpy(
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

        const int totalWeights =
            bufferRows * bufferCols;

        assert(srcStart + length <= totalWeights);

        CUDA_CHECK_THROW(cudaMemcpy(
            dst + dstStart,
            weightsBuffer + srcStart,
            static_cast<size_t>(length) *
            sizeof(float),
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

        CUDA_CHECK_THROW(cudaMemcpy(
            dst + dstStart,
            outputBuffer + srcStart,
            static_cast<size_t>(length) *
            sizeof(float),
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

        CUDA_CHECK_THROW(cudaGetLastError());
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
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

        CUDA_CHECK_THROW(cudaMemcpy(
            inputBuffer,
            input,
            static_cast<size_t>(cols) *
            sizeof(float),
            cudaMemcpyHostToDevice
        ));

        launchMatmul(
            inputBuffer,
            outputBuffer,
            rows,
            cols
        );

        CUDA_CHECK_THROW(cudaMemcpy(
            output,
            outputBuffer,
            static_cast<size_t>(rows) *
            sizeof(float),
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

        CUDA_CHECK_THROW(cudaMemcpy(
            output,
            outputBuffer,
            static_cast<size_t>(rows) *
            sizeof(float),
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

        CUDA_CHECK_THROW(cudaMemcpy(
            inputBuffer,
            input,
            static_cast<size_t>(cols) *
            sizeof(float),
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

    // ============================================================
    // INTERNAL LOGITS LAUNCH
    // ============================================================

    void launchComputeLogits(
        float* x,
        float* logits,
        int vocabSize,
        int hiddenSize
    ) {
        constexpr int THREADS = 256;

        const int blocks =
            (vocabSize + THREADS - 1) / THREADS;

        compute_logits_kernel<<<blocks, THREADS>>>(
            x,
            weightsBuffer,
            logits,
            static_cast<uint32_t>(vocabSize),
            static_cast<uint32_t>(hiddenSize)
        );

        CUDA_CHECK_THROW(cudaGetLastError());
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }

    // ============================================================
    // COMPUTE LOGITS
    // ============================================================

    void computeLogits(
        const float* x,
        float* logits,
        int vocabSize,
        int hiddenSize
    ) {
        assert(vocabSize > 0);
        assert(hiddenSize > 0);
        assert(weightsLoaded);

        ensureBuffers(vocabSize, hiddenSize);

        CUDA_CHECK_THROW(cudaMemcpy(
            inputBuffer,
            x,
            static_cast<size_t>(hiddenSize) *
            sizeof(float),
            cudaMemcpyHostToDevice
        ));

        launchComputeLogits(
            inputBuffer,
            outputBuffer,
            vocabSize,
            hiddenSize
        );

        CUDA_CHECK_THROW(cudaMemcpy(
            logits,
            outputBuffer,
            static_cast<size_t>(vocabSize) *
            sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }
};

// ============================================================
// C API
// ============================================================

extern "C" {

// ============================================================
// CREATE / DESTROY
// ============================================================

__declspec(dllexport)
void* cuda_create() {

    try {
        return new CudaBackend();
    } catch (...) {
        return nullptr;
    }
}

__declspec(dllexport)
void cuda_destroy(void* ptr) {

    if (ptr == nullptr) {
        return;
    }

    delete static_cast<CudaBackend*>(ptr);
}

// ============================================================
// SET WEIGHTS
// ============================================================

__declspec(dllexport)
void cuda_set_weights(
    void* ptr,
    const float* weights,
    int32_t rows,
    int32_t cols
) {
    if (ptr == nullptr) {
        return;
    }

    try {

        auto* backend =
            static_cast<CudaBackend*>(ptr);

        backend->setWeights(
            weights,
            static_cast<int>(rows),
            static_cast<int>(cols)
        );

    } catch (...) {
        return;
    }
}

// ============================================================
// COPY WEIGHTS
// ============================================================

__declspec(dllexport)
void cuda_copy_weights(
    void* ptr,
    float* dst,
    int dstStart,
    int srcStart,
    int length
) {
    if (ptr == nullptr) {
        return;
    }

    try {

        auto* backend =
            static_cast<CudaBackend*>(ptr);

        backend->copyWeights(
            dst,
            dstStart,
            srcStart,
            length
        );

    } catch (...) {
        return;
    }
}

// ============================================================
// COPY OUTPUT
// ============================================================

__declspec(dllexport)
void cuda_copy_output(
    void* ptr,
    float* dst,
    int dstStart,
    int srcStart,
    int length
) {
    if (ptr == nullptr) {
        return;
    }

    try {

        auto* backend =
            static_cast<CudaBackend*>(ptr);

        backend->copyOutput(
            dst,
            dstStart,
            srcStart,
            length
        );

    } catch (...) {
        return;
    }
}

// ============================================================
// MATMUL
// ============================================================

__declspec(dllexport)
void cuda_matmul(
    void* ptr,
    const float* input,
    float* output,
    int rows,
    int cols
) {
    if (ptr == nullptr) {
        return;
    }

    try {

        auto* backend =
            static_cast<CudaBackend*>(ptr);

        backend->matmul(
            input,
            output,
            rows,
            cols
        );

    } catch (...) {
        return;
    }
}

// ============================================================
// GPU INPUT BUFFER
// ============================================================

__declspec(dllexport)
void cuda_matmul_input_cudabuffer(
    void* ptr,
    void* inputBuffer,
    float* output,
    int rows,
    int cols
) {
    if (ptr == nullptr) {
        return;
    }

    try {

        auto* backend =
            static_cast<CudaBackend*>(ptr);

        backend->matmulInputCudaBuffer(
            static_cast<float*>(inputBuffer),
            output,
            rows,
            cols
        );

    } catch (...) {
        return;
    }
}

// ============================================================
// GPU OUTPUT BUFFER
// ============================================================

__declspec(dllexport)
void cuda_matmul_output_cudabuffer(
    void* ptr,
    const float* input,
    void* outputBuffer,
    int rows,
    int cols
) {
    if (ptr == nullptr) {
        return;
    }

    try {

        auto* backend =
            static_cast<CudaBackend*>(ptr);

        backend->matmulOutputCudaBuffer(
            input,
            static_cast<float*>(outputBuffer),
            rows,
            cols
        );

    } catch (...) {
        return;
    }
}

// ============================================================
// GPU INPUT + OUTPUT BUFFERS
// ============================================================

__declspec(dllexport)
void cuda_matmul_input_output_cudabuffer(
    void* ptr,
    void* inputBuffer,
    void* outputBuffer,
    int rows,
    int cols
) {
    if (ptr == nullptr) {
        return;
    }

    try {

        auto* backend =
            static_cast<CudaBackend*>(ptr);

        backend->matmulInputOutputCudaBuffer(
            static_cast<float*>(inputBuffer),
            static_cast<float*>(outputBuffer),
            rows,
            cols
        );

    } catch (...) {
        return;
    }
}

// ============================================================
// BUFFER CREATE / DESTROY
// ============================================================

__declspec(dllexport)
void* cuda_create_float_buffer(int size) {

    float* ptr = nullptr;

    CUDA_CHECK_RETURN_NULL(cudaMalloc(
        reinterpret_cast<void**>(&ptr),
        static_cast<size_t>(size) *
        sizeof(float)
    ));

    return ptr;
}

__declspec(dllexport)
void cuda_destroy_buffer(void* ptr) {

    if (ptr == nullptr) {
        return;
    }

    CUDA_CHECK_RETURN(cudaFree(ptr));
}

// ============================================================
// SET RANGE
// ============================================================

__declspec(dllexport)
void cuda_set_range(
    void* ptr,
    const float* src,
    int dstStart,
    int srcStart,
    int count
) {
    CUDA_CHECK_RETURN(cudaMemcpy(
        static_cast<float*>(ptr) + dstStart,
        src + srcStart,
        static_cast<size_t>(count) *
        sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

// ============================================================
// COPY DATA TO
// ============================================================

__declspec(dllexport)
void cuda_copy_data_to(
    void* ptr,
    float* dst,
    int srcStart,
    int dstStart,
    int count
) {
    CUDA_CHECK_RETURN(cudaMemcpy(
        dst + dstStart,
        static_cast<float*>(ptr) + srcStart,
        static_cast<size_t>(count) *
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

// ============================================================
// ADD GPU BUFFER TO CPU BUFFER
// ============================================================

__declspec(dllexport)
void add_gpu_buffer_to_cpu_buffer(
    void* gpuBuffer,
    float* cpuBuffer,
    int length
) {
    float* tmp = new float[length];

    cudaError_t err = cudaMemcpy(
        tmp,
        gpuBuffer,
        static_cast<size_t>(length) *
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess) {
        delete[] tmp;
        return;
    }

    for (int i = 0; i < length; i++) {
        cpuBuffer[i] += tmp[i];
    }

    delete[] tmp;
}

}