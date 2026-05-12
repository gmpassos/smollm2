import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import '../../native_tensor.dart';

class CudaLibrary extends NativeLibrary<CudaBinding> {
  static final instance = CudaLibrary._();

  CudaLibrary._();

  factory CudaLibrary() => instance;

  @override
  String get name => 'CUDA';

  @override
  File? tryResolveLibraryFile() {
    return File(r'C:\smollm2\native\windows\cuda\build\cuda_backend.dll');
  }

  @override
  Set<NativeTensorReader> tensorReaders = const {};

  @override
  CudaBinding create() {
    final ptr = cudaCreate();
    return CudaBinding._(ptr);
  }

  @Native<Pointer<Void> Function()>(symbol: 'cuda_create', isLeaf: true)
  external static Pointer<Void> cudaCreate();

  @Native<Void Function(Pointer<Void>)>(symbol: 'cuda_destroy', isLeaf: true)
  external static void cudaDestroy(Pointer<Void> ptr);

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int32, Int32)>(
    symbol: 'cuda_set_weights',
    isLeaf: true,
  )
  external static void cudaSetWeights(
    Pointer<Void> backend,
    Pointer<Float> weights,
    int rows,
    int cols,
  );

  @Native<
    Void Function(Pointer<Void>, Pointer<Float>, Pointer<Float>, Int, Int)
  >(symbol: 'cuda_matmul', isLeaf: true)
  external static void cudaMatmul(
    Pointer<Void> backend,
    Pointer<Float> input,
    Pointer<Float> output,
    int rows,
    int cols,
  );

  @Native<
    Void Function(Pointer<Void>, Pointer<Void>, Pointer<Float>, Int, Int)
  >(symbol: 'cuda_matmul_input_cudabuffer', isLeaf: true)
  external static void cudaMatmulInputCudaBuffer(
    Pointer<Void> backend,
    Pointer<Void> inputBuffer,
    Pointer<Float> output,
    int rows,
    int cols,
  );

  @Native<
    Void Function(Pointer<Void>, Pointer<Float>, Pointer<Void>, Int, Int)
  >(symbol: 'cuda_matmul_output_cudabuffer', isLeaf: true)
  external static void cudaMatmulOutputCudaBuffer(
    Pointer<Void> backend,
    Pointer<Float> input,
    Pointer<Void> outputBuffer,
    int rows,
    int cols,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Void>, Pointer<Void>, Int, Int)>(
    symbol: 'cuda_matmul_input_output_cudabuffer',
    isLeaf: true,
  )
  external static void cudaMatmulInputOutputCudaBuffer(
    Pointer<Void> backend,
    Pointer<Void> inputBuffer,
    Pointer<Void> outputBuffer,
    int rows,
    int cols,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int, Int, Int)>(
    symbol: 'cuda_copy_weights',
    isLeaf: true,
  )
  external static void cudaCopyWeights(
    Pointer<Void> backend,
    Pointer<Float> dst,
    int dstStart,
    int srcStart,
    int length,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int, Int, Int)>(
    symbol: 'cuda_copy_output',
    isLeaf: true,
  )
  external static void cudaCopyOutput(
    Pointer<Void> backend,
    Pointer<Float> dst,
    int dstStart,
    int srcStart,
    int length,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int)>(
    symbol: 'add_gpu_buffer_to_cpu_buffer',
    isLeaf: true,
  )
  external static void addGpuBufferToCpuBuffer(
    Pointer<Void> gpuBuffer,
    Pointer<Float> cpuBuffer,
    int length,
  );

  @Native<Pointer<Void> Function(Int)>(
    symbol: 'cuda_create_float_buffer',
    isLeaf: true,
  )
  external static Pointer<Void> cudaCreateFloatBuffer(int length);

  @Native<Void Function(Pointer<Void>)>(
    symbol: 'cuda_destroy_buffer',
    isLeaf: true,
  )
  external static void cudaDestroyBuffer(Pointer<Void> buffer);

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int, Int, Int)>(
    symbol: 'cuda_set_range',
    isLeaf: true,
  )
  external static void cudaSetRange(
    Pointer<Void> buffer,
    Pointer<Float> src,
    int dstStart,
    int srcStart,
    int count,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int, Int, Int)>(
    symbol: 'cuda_copy_data_to',
    isLeaf: true,
  )
  external static void cudaCopyDataTo(
    Pointer<Void> buffer,
    Pointer<Float> dst,
    int srcStart,
    int dstStart,
    int count,
  );
}

class CudaBinding implements NativeBinding {
  final Pointer<Void> _ptr;

  CudaBinding._(this._ptr);

  void setWeights(Float32List weights, int rows, int cols) {
    assert(CudaLibrary.instance.isLibraryLoaded);

    CudaLibrary.cudaSetWeights(_ptr, weights.address, rows, cols);
  }

  void cudaMatmul(Float32List input, Float32List output, int rows, int cols) {
    assert(CudaLibrary.instance.isLibraryLoaded);

    CudaLibrary.cudaMatmul(_ptr, input.address, output.address, rows, cols);
  }

  void cudaMatmulInputCudaBuffer(
    Pointer<Void> inputBuffer,
    Float32List output,
    int rows,
    int cols,
  ) {
    assert(CudaLibrary.instance.isLibraryLoaded);

    CudaLibrary.cudaMatmulInputCudaBuffer(
      _ptr,
      inputBuffer,
      output.address,
      rows,
      cols,
    );
  }

  void cudaMatmulOutputCudaBuffer(
    Float32List input,
    Pointer<Void> outputBuffer,
    int rows,
    int cols,
  ) {
    assert(CudaLibrary.instance.isLibraryLoaded);

    CudaLibrary.cudaMatmulOutputCudaBuffer(
      _ptr,
      input.address,
      outputBuffer,
      rows,
      cols,
    );
  }

  void cudaMatmulInputOutputCudaBuffer(
    Pointer<Void> inputBuffer,
    Pointer<Void> outputBuffer,
    int rows,
    int cols,
  ) {
    assert(CudaLibrary.instance.isLibraryLoaded);

    CudaLibrary.cudaMatmulInputOutputCudaBuffer(
      _ptr,
      inputBuffer,
      outputBuffer,
      rows,
      cols,
    );
  }

  void copyWeightsTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    assert(CudaLibrary.instance.isLibraryLoaded);

    final length = dstEnd - dstStart;

    CudaLibrary.cudaCopyWeights(_ptr, dst.address, dstStart, dataStart, length);
  }

  void copyOutputTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    assert(CudaLibrary.instance.isLibraryLoaded);

    final length = dstEnd - dstStart;

    CudaLibrary.cudaCopyOutput(_ptr, dst.address, dstStart, dataStart, length);
  }

  void dispose() {
    assert(CudaLibrary.instance.isLibraryLoaded);
    CudaLibrary.cudaDestroy(_ptr);
  }
}

class CudaFloat32Buffer {
  static final Finalizer<Pointer<Void>> _finalizer = Finalizer<Pointer<Void>>((
    ptr,
  ) {
    CudaLibrary.cudaDestroyBuffer(ptr);
  });

  final Pointer<Void> _ptr;

  final int length;

  bool _disposed = false;

  CudaFloat32Buffer._(this._ptr, this.length) {
    _finalizer.attach(this, _ptr, detach: this);
  }

  factory CudaFloat32Buffer(int length) {
    assert(CudaLibrary.instance.isLibraryLoaded);

    final ptr = CudaLibrary.cudaCreateFloatBuffer(length);

    return CudaFloat32Buffer._(ptr, length);
  }

  Pointer<Void> get ptr => _ptr;

  bool get isDisposed => _disposed;

  void setRange(int start, int end, Float32List iterable, [int skipCount = 0]) {
    if (_disposed) {
      throw StateError('Buffer already disposed');
    }

    if (start < 0 || end < start || end > length) {
      throw RangeError.range(start, 0, length);
    }

    final count = end - start;

    if (skipCount < 0 || skipCount + count > iterable.length) {
      throw RangeError('Invalid skipCount or iterable too small');
    }

    CudaLibrary.cudaSetRange(_ptr, iterable.address, start, skipCount, count);
  }

  void copyDataTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    if (_disposed) {
      throw StateError('Buffer already disposed');
    }

    if (dstStart < 0 || dstEnd < dstStart || dstEnd > dst.length) {
      throw RangeError.range(dstStart, 0, dst.length);
    }

    final count = dstEnd - dstStart;

    if (dataStart < 0 || dataStart + count > length) {
      throw RangeError('Invalid dataStart or buffer too small');
    }

    CudaLibrary.cudaCopyDataTo(_ptr, dst.address, dataStart, dstStart, count);
  }

  void addTo(Float32List dst, int length) {
    if (_disposed) {
      throw StateError('Buffer already disposed');
    }

    if (length > this.length) {
      throw RangeError.range(
        length,
        0,
        this.length,
        'length',
        'Must be ≤ ${this.length}',
      );
    }

    CudaLibrary.addGpuBufferToCpuBuffer(_ptr, dst.address, length);
  }

  void dispose() {
    if (_disposed) {
      return;
    }

    assert(CudaLibrary.instance.isLibraryLoaded);

    _disposed = true;

    _finalizer.detach(this);

    CudaLibrary.cudaDestroyBuffer(_ptr);
  }

  @override
  String toString() {
    return 'CudaFloat32Buffer#${_ptr.address}{length: $length}';
  }
}
