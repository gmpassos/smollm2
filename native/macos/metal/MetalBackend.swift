import Foundation
import Metal
import simd

// MARK: - Metal Context

final class MetalContext {

    static let sharedDevice: MTLDevice = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not available")
        }
        return device
    }()

    static let sharedQueue: MTLCommandQueue = {
        guard let queue = sharedDevice.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        return queue
    }()

    static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void matmul(
        const device float* weights [[buffer(0)]],
        const device float* input   [[buffer(1)]],
        device float* output        [[buffer(2)]],
        constant uint& cols         [[buffer(3)]],
        uint gid                    [[thread_position_in_grid]]
    ) {
        float sum = 0.0;
        uint rowOffset = gid * cols;

        uint simdCols = cols / 4;

        const device float4* weights4 =
            reinterpret_cast<const device float4*>(weights + rowOffset);

        const device float4* input4 =
            reinterpret_cast<const device float4*>(input);

        for (uint i = 0; i < simdCols; i++) {
            sum += dot(weights4[i], input4[i]);
        }

        uint remainderStart = simdCols * 4;

        for (uint i = remainderStart; i < cols; i++) {
            sum += weights[rowOffset + i] * input[i];
        }

        output[gid] = sum;
    }

    kernel void compute_logits(
        const device float* x         [[buffer(0)]],
        const device float* embed     [[buffer(1)]],
        device float* logits          [[buffer(2)]],
        constant uint& hiddenSize     [[buffer(3)]],
        uint gid                      [[thread_position_in_grid]]
    ) {
        float sum = 0.0;

        uint rowOffset = gid * hiddenSize;

        uint simdCols = hiddenSize / 4;

        const device float4* x4 =
            reinterpret_cast<const device float4*>(x);

        const device float4* embed4 =
            reinterpret_cast<const device float4*>(embed + rowOffset);

        for (uint i = 0; i < simdCols; i++) {
            sum += dot(x4[i], embed4[i]);
        }

        uint tailStart = simdCols * 4;

        for (uint i = tailStart; i < hiddenSize; i++) {
            sum += x[i] * embed[rowOffset + i];
        }

        logits[gid] = sum;
    }
    """

    static let sharedPipeline: MTLComputePipelineState = {
        do {
            let library = try sharedDevice.makeLibrary(
                source: metalSource,
                options: nil
            )

            guard let function = library.makeFunction(name: "matmul") else {
                fatalError("Failed to create function")
            }

            return try sharedDevice.makeComputePipelineState(
                function: function
            )

        } catch {
            fatalError("Pipeline error: \\(error)")
        }
    }()

    static let sharedLogitsPipeline: MTLComputePipelineState = {
        do {
            let library = try sharedDevice.makeLibrary(
                source: metalSource,
                options: nil
            )

            guard let function = library.makeFunction(
                name: "compute_logits"
            ) else {
                fatalError("Failed to create compute_logits")
            }

            return try sharedDevice.makeComputePipelineState(
                function: function
            )

        } catch {
            fatalError("Pipeline error: \\(error)")
        }
    }()
}

// MARK: - Backend

final class MetalBackend {

    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    let logitsPipeline: MTLComputePipelineState

    private var weightsBuffer: MTLBuffer?
    private var inputBuffer: MTLBuffer?
    private var outputBuffer: MTLBuffer?
    private var colsBuffer: MTLBuffer?

    private var logitsBuffer: MTLBuffer?
    private var hiddenSizeBuffer: MTLBuffer?

    // Store current dimensions
    private var bufferRows: Int = 0
    private var bufferCols: Int = 0

    private var weightsLoaded = false

    init() {
        self.device = MetalContext.sharedDevice
        self.queue = MetalContext.sharedQueue
        self.pipeline = MetalContext.sharedPipeline
        self.logitsPipeline = MetalContext.sharedLogitsPipeline
    }

    private func ensureBuffers(rows: Int, cols: Int) {
        // Fast path
        if bufferRows >= rows &&
           bufferCols >= cols &&
           weightsBuffer != nil &&
           inputBuffer != nil &&
           outputBuffer != nil &&
           colsBuffer != nil &&
           logitsBuffer != nil &&
           hiddenSizeBuffer != nil {
            return
        }

        let weightsSize = rows * cols * MemoryLayout<Float>.stride
        let inputSize = cols * MemoryLayout<Float>.stride
        let outputSize = rows * MemoryLayout<Float>.stride
        let colsSize = MemoryLayout<UInt32>.stride

        let logitsSize = rows * MemoryLayout<Float>.stride
        let hiddenSizeSize = MemoryLayout<UInt32>.stride

        if weightsBuffer == nil || weightsBuffer!.length < weightsSize {
            weightsBuffer = device.makeBuffer(
                length: weightsSize,
                options: .storageModeShared
            )
        }

        if inputBuffer == nil || inputBuffer!.length < inputSize {
            inputBuffer = device.makeBuffer(
                length: inputSize,
                options: .storageModeShared
            )
        }

        if outputBuffer == nil || outputBuffer!.length < outputSize {
            outputBuffer = device.makeBuffer(
                length: outputSize,
                options: .storageModeShared
            )
        }

        if colsBuffer == nil || colsBuffer!.length < colsSize {
            colsBuffer = device.makeBuffer(
                length: colsSize,
                options: .storageModeShared
            )
        }

        if logitsBuffer == nil || logitsBuffer!.length < logitsSize {
            logitsBuffer = device.makeBuffer(
                length: logitsSize,
                options: .storageModeShared
            )
        }

        if hiddenSizeBuffer == nil ||
            hiddenSizeBuffer!.length < hiddenSizeSize {

            hiddenSizeBuffer = device.makeBuffer(
                length: hiddenSizeSize,
                options: .storageModeShared
            )
        }

        // Save allocated capacity
        bufferRows = max(bufferRows, rows)
        bufferCols = max(bufferCols, cols)
    }

    // MARK: - WEIGHTS

    func setWeights(_ weights: UnsafePointer<Float>, rows: Int, cols: Int) {
        ensureBuffers(rows: rows, cols: cols)

        let size = rows * cols * MemoryLayout<Float>.stride

        weightsBuffer!.contents().copyMemory(
            from: weights,
            byteCount: size
        )

        weightsLoaded = true
    }

    // MARK: - WEIGHTS COPY

    func copyWeights(
        to dst: UnsafeMutablePointer<Float>,
        dstStart: Int,
        srcStart: Int,
        length: Int
    ) {
        precondition(length >= 0)
        precondition(dstStart >= 0)
        precondition(srcStart >= 0)

        let totalWeights = bufferRows * bufferCols

        precondition(
            srcStart + length <= totalWeights,
            """
            copyWeights OUT OF BOUNDS:
              requested elements: \(srcStart)..<\(srcStart + length)

              bufferRows: \(bufferRows)
              bufferCols: \(bufferCols)

              valid element range: 0..<\(totalWeights)
            """
        )

        guard let weightsBuffer else {
            print("copyWeights ERROR: weightsBuffer == nil")
            return
        }

        let stride = MemoryLayout<Float>.stride

        let srcOffset = srcStart * stride
        let dstOffset = dstStart * stride
        let copyBytes = length * stride

        let srcRangeEnd = srcOffset + copyBytes

        precondition(
            srcRangeEnd <= weightsBuffer.length,
            """
            copyWeights OUT OF BOUNDS:
              srcRangeEnd (\(srcRangeEnd))
              > weightsBuffer.length (\(weightsBuffer.length))
            """
        )

        let src = weightsBuffer.contents()
            .advanced(by: srcOffset)

        let dst2 = UnsafeMutableRawPointer(dst)
            .advanced(by: dstOffset)

        memcpy(dst2, src, copyBytes)
    }

    func copyOutput(
        to dst: UnsafeMutablePointer<Float>,
        dstStart: Int,
        srcStart: Int,
        length: Int
    ) {
        precondition(length >= 0)
        precondition(dstStart >= 0)
        precondition(srcStart >= 0)

        precondition(
            srcStart + length <= bufferRows,
            """
            copyOutput OUT OF BOUNDS:
              requested rows: \(srcStart)..<\(srcStart + length)

              bufferRows: \(bufferRows)
              bufferCols: \(bufferCols)

              valid row range: 0..<\(bufferRows)
              valid element capacity: \(bufferRows * bufferCols)
            """
        )

        guard let outputBuffer else {
            print("copyOutput ERROR: outputBuffer == nil")
            return
        }

        let stride = MemoryLayout<Float>.stride

        let srcOffset = srcStart * stride
        let dstOffset = dstStart * stride
        let copyBytes = length * stride

        let srcRangeEnd = srcOffset + copyBytes

        precondition(
            srcRangeEnd <= outputBuffer.length,
            """
            copyOutput OUT OF BOUNDS:
              srcRangeEnd (\(srcRangeEnd))
              > outputBuffer.length (\(outputBuffer.length))
            """
        )

        let src = outputBuffer.contents()
            .advanced(by: srcOffset)

        let dst2 = UnsafeMutableRawPointer(dst)
            .advanced(by: dstOffset)

        memcpy(dst2, src, copyBytes)
    }

    // MARK: - MATMUL

    func matmul(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        rows: Int,
        cols: Int
    ) {
        precondition(weightsLoaded, "Call setWeights before matmul")
        precondition(rows > 0 && cols > 0, "Invalid rows/cols")

        guard let weightsBuffer,
              let inputBuffer,
              let outputBuffer,
              let colsBuffer else {
            fatalError("Metal buffers not initialized")
        }

        let inputSize = cols * MemoryLayout<Float>.stride

        inputBuffer.contents().copyMemory(
            from: input,
            byteCount: inputSize
        )

        var colsValue = UInt32(cols)

        colsBuffer.contents().copyMemory(
            from: &colsValue,
            byteCount: MemoryLayout<UInt32>.stride
        )

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBuffer(colsBuffer, offset: 0, index: 3)

        let width = pipeline.threadExecutionWidth
        let tgWidth = min(width, rows)

        encoder.dispatchThreads(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: tgWidth,
                height: 1,
                depth: 1
            )
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // GPU → CPU copy
        let outputSize = rows * MemoryLayout<Float>.stride

        memcpy(
            output,
            outputBuffer.contents(),
            outputSize
        )
    }

    func matmulInputMLTBuffer(
        inputBuffer input: MTLBuffer,
        output: UnsafeMutablePointer<Float>,
        rows: Int,
        cols: Int
    ) {
        precondition(weightsLoaded, "Call setWeights before matmul")
        precondition(rows > 0 && cols > 0)

        guard let weightsBuffer,
              let colsBuffer,
              let outputBuffer,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            fatalError("Metal buffers not initialized")
        }

        // upload cols
        var colsValue = UInt32(cols)

        colsBuffer.contents().copyMemory(
            from: &colsValue,
            byteCount: MemoryLayout<UInt32>.stride
        )

        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBuffer(colsBuffer, offset: 0, index: 3)

        let width = pipeline.threadExecutionWidth
        let tgWidth = min(width, rows)

        encoder.dispatchThreads(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: tgWidth,
                height: 1,
                depth: 1
            )
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // GPU → CPU copy
        let outputSize = rows * MemoryLayout<Float>.stride

        memcpy(
            output,
            outputBuffer.contents(),
            outputSize
        )
    }

    func matmulOutputMLTBuffer(
        input: UnsafePointer<Float>,
        outputBuffer: MTLBuffer,
        rows: Int,
        cols: Int
    ) {
        precondition(weightsLoaded, "Call setWeights before matmul")
        precondition(rows > 0 && cols > 0)

        guard let weightsBuffer,
              let inputBuffer,
              let colsBuffer,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            fatalError("Metal buffers not initialized")
        }

        // CPU → GPU (input copy)
        let inputSize = cols * MemoryLayout<Float>.stride

        inputBuffer.contents().copyMemory(
            from: input,
            byteCount: inputSize
        )

        // upload cols
        var colsValue = UInt32(cols)

        colsBuffer.contents().copyMemory(
            from: &colsValue,
            byteCount: MemoryLayout<UInt32>.stride
        )

        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBuffer(colsBuffer, offset: 0, index: 3)

        let width = pipeline.threadExecutionWidth
        let tgWidth = min(width, rows)

        encoder.dispatchThreads(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: tgWidth,
                height: 1,
                depth: 1
            )
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func matmulInputOutputMLTBuffer(
        inputBuffer input: MTLBuffer,
        outputBuffer output: MTLBuffer,
        rows: Int,
        cols: Int
    ) {
        precondition(weightsLoaded, "Call setWeights before matmul")
        precondition(rows > 0 && cols > 0)

        guard let weightsBuffer,
              let colsBuffer,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            fatalError("Metal buffers not initialized")
        }

        // upload cols
        var colsValue = UInt32(cols)

        colsBuffer.contents().copyMemory(
            from: &colsValue,
            byteCount: MemoryLayout<UInt32>.stride
        )

        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(colsBuffer, offset: 0, index: 3)

        let width = pipeline.threadExecutionWidth
        let tgWidth = min(width, rows)

        encoder.dispatchThreads(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: tgWidth,
                height: 1,
                depth: 1
            )
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func computeLogits(
        x: UnsafePointer<Float>,
        logits: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        hiddenSize: Int
    ) {
        precondition(vocabSize > 0)
        precondition(hiddenSize > 0)

        ensureBuffers(rows: vocabSize, cols: hiddenSize)

        guard let inputBuffer,
              let logitsBuffer,
              let hiddenSizeBuffer,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            fatalError("Metal buffers not initialized")
        }

        let embedBuffer = weightsBuffer;

        let inputSize = hiddenSize * MemoryLayout<Float>.stride

        inputBuffer.contents().copyMemory(
            from: x,
            byteCount: inputSize
        )

        var hs = UInt32(hiddenSize)

        hiddenSizeBuffer.contents().copyMemory(
            from: &hs,
            byteCount: MemoryLayout<UInt32>.stride
        )

        encoder.setComputePipelineState(logitsPipeline)

        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(embedBuffer, offset: 0, index: 1)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 2)
        encoder.setBuffer(hiddenSizeBuffer, offset: 0, index: 3)

        let width = logitsPipeline.threadExecutionWidth
        let tgWidth = min(width, vocabSize)

        encoder.dispatchThreads(
            MTLSize(width: vocabSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: tgWidth,
                height: 1,
                depth: 1
            )
        )

        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        memcpy(
            logits,
            logitsBuffer.contents(),
            vocabSize * MemoryLayout<Float>.stride
        )
    }
}

// MARK: - C API

@_cdecl("metal_create")
public func metal_create() -> UnsafeMutableRawPointer {
    let backend = MetalBackend()
    return Unmanaged.passRetained(backend).toOpaque()
}

@_cdecl("metal_destroy")
public func metal_destroy(_ ptr: UnsafeMutableRawPointer) {
    Unmanaged<MetalBackend>.fromOpaque(ptr).release()
}

@_cdecl("metal_set_weights")
public func metal_set_weights(
    _ ptr: UnsafeMutableRawPointer,
    _ weights: UnsafePointer<Float>,
    _ rows: Int32,
    _ cols: Int32
) {
    let backend = Unmanaged<MetalBackend>.fromOpaque(ptr)
        .takeUnretainedValue()

    backend.setWeights(
        weights,
        rows: Int(rows),
        cols: Int(cols)
    )
}

@_cdecl("metal_copy_weights")
public func metal_copy_weights(
    _ ptr: UnsafeMutableRawPointer,
    _ dst: UnsafeMutablePointer<Float>,
    _ dstStart: Int,
    _ srcStart: Int,
    _ length: Int
) {
    precondition(length >= 0)
    precondition(dstStart >= 0)
    precondition(srcStart >= 0)

    let backend = Unmanaged<MetalBackend>
        .fromOpaque(ptr)
        .takeUnretainedValue()

    backend.copyWeights(
        to: dst,
        dstStart: dstStart,
        srcStart: srcStart,
        length: length
    )
}

@_cdecl("metal_copy_output")
public func metal_copy_output(
    _ ptr: UnsafeMutableRawPointer,
    _ dst: UnsafeMutablePointer<Float>,
    _ dstStart: Int,
    _ srcStart: Int,
    _ length: Int
) {
    precondition(length >= 0)
    precondition(dstStart >= 0)
    precondition(srcStart >= 0)

    let backend = Unmanaged<MetalBackend>
        .fromOpaque(ptr)
        .takeUnretainedValue()

    backend.copyOutput(
        to: dst,
        dstStart: dstStart,
        srcStart: srcStart,
        length: length
    )
}

@_cdecl("metal_matmul")
public func metal_matmul(
    _ ptr: UnsafeMutableRawPointer,
    _ input: UnsafePointer<Float>,
    _ output: UnsafeMutablePointer<Float>,
    _ rows: Int,
    _ cols: Int
) {
    let backend = Unmanaged<MetalBackend>.fromOpaque(ptr)
        .takeUnretainedValue()

    backend.matmul(
        input: input,
        output: output,
        rows: rows,
        cols: cols
    )
}

@_cdecl("metal_matmul_input_mltbuffer")
public func metal_matmul_input_mltbuffer(
    _ ptr: UnsafeMutableRawPointer,
    _ inputBuf: UnsafeMutableRawPointer,
    _ output: UnsafeMutablePointer<Float>,
    _ rows: Int,
    _ cols: Int
) {
    let backend = Unmanaged<MetalBackend>
        .fromOpaque(ptr)
        .takeUnretainedValue()

    let input = Unmanaged<MTLBuffer>
        .fromOpaque(inputBuf)
        .takeUnretainedValue()

    backend.matmulInputMLTBuffer(
        inputBuffer: input,
        output: output,
        rows: rows,
        cols: cols
    )
}

@_cdecl("metal_matmul_output_mltbuffer")
public func metal_matmul_output_mltbuffer(
    _ ptr: UnsafeMutableRawPointer,
    _ input: UnsafePointer<Float>,
    _ outputBuf: UnsafeMutableRawPointer,
    _ rows: Int,
    _ cols: Int
) {
    let backend = Unmanaged<MetalBackend>
        .fromOpaque(ptr)
        .takeUnretainedValue()

    let output = Unmanaged<MTLBuffer>
        .fromOpaque(outputBuf)
        .takeUnretainedValue()

    backend.matmulOutputMLTBuffer(
        input: input,
        outputBuffer: output,
        rows: rows,
        cols: cols
    )
}

@_cdecl("metal_matmul_input_output_mltbuffer")
public func metal_matmul_input_output_mltbuffer(
    _ ptr: UnsafeMutableRawPointer,
    _ inputBuf: UnsafeMutableRawPointer,
    _ outputBuf: UnsafeMutableRawPointer,
    _ rows: Int,
    _ cols: Int
) {
    let backend = Unmanaged<MetalBackend>
        .fromOpaque(ptr)
        .takeUnretainedValue()

    let input = Unmanaged<MTLBuffer>
        .fromOpaque(inputBuf)
        .takeUnretainedValue()

    let output = Unmanaged<MTLBuffer>
        .fromOpaque(outputBuf)
        .takeUnretainedValue()

    backend.matmulInputOutputMLTBuffer(
        inputBuffer: input,
        outputBuffer: output,
        rows: rows,
        cols: cols
    )
}

@_cdecl("metal_compute_logits")
public func metal_compute_logits(
    _ ptr: UnsafeMutableRawPointer,
    _ x: UnsafePointer<Float>,
    _ logits: UnsafeMutablePointer<Float>,
    _ vocabSize: Int,
    _ hiddenSize: Int
) {
    let backend = Unmanaged<MetalBackend>
        .fromOpaque(ptr)
        .takeUnretainedValue()

    backend.computeLogits(
        x: x,
        logits: logits,
        vocabSize: vocabSize,
        hiddenSize: hiddenSize
    )
}

@_cdecl("add_gpu_buffer_to_cpu_buffer")
public func add_gpu_buffer_to_cpu_buffer(
    _ gpuBuffer: MTLBuffer,
    _ cpuBuffer: UnsafeMutablePointer<Float>,
    _ length: Int
) {
    let gpuPtr = gpuBuffer.contents()
        .bindMemory(to: Float.self, capacity: length)

    for i in 0..<length {
        cpuBuffer[i] += gpuPtr[i]
    }
}

@_cdecl("metal_create_float_buffer")
public func metal_create_float_buffer(
    _ size: Int
) -> UnsafeMutableRawPointer {

    let device = MetalContext.sharedDevice

    guard let buffer = device.makeBuffer(
        length: size * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ) else {
        fatalError("Failed to create buffer")
    }

    return Unmanaged.passRetained(buffer).toOpaque()
}

@_cdecl("metal_destroy_buffer")
public func metal_destroy_buffer(_ ptr: UnsafeMutableRawPointer) {
    let buffer = Unmanaged<MTLBuffer>.fromOpaque(ptr)
    buffer.release()
}

@_cdecl("metal_set_range")
public func metal_set_range(
    _ ptr: UnsafeMutableRawPointer,
    _ src: UnsafePointer<Float>,
    _ dstStart: Int,
    _ srcStart: Int,
    _ count: Int
) {
    let buffer = Unmanaged<MTLBuffer>
        .fromOpaque(ptr)
        .takeUnretainedValue()

    let dstBase = buffer.contents()
        .assumingMemoryBound(to: Float.self)

    let dst = dstBase.advanced(by: dstStart)
    let src = src.advanced(by: srcStart)

    memcpy(
        dst,
        src,
        count * MemoryLayout<Float>.stride
    )
}

@_cdecl("metal_copy_data_to")
public func metal_copy_data_to(
    _ ptr: UnsafeMutableRawPointer,
    _ dst: UnsafeMutablePointer<Float>,
    _ srcStart: Int,
    _ dstStart: Int,
    _ count: Int
) {
    let buffer = Unmanaged<MTLBuffer>
        .fromOpaque(ptr)
        .takeUnretainedValue()

    let srcBase = buffer.contents()
        .assumingMemoryBound(to: Float.self)

    let src = srcBase.advanced(by: srcStart)
    let dst = dst.advanced(by: dstStart)

    memcpy(
        dst,
        src,
        count * MemoryLayout<Float>.stride
    )
}