import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import '../../native_tensor.dart';
import 'fp32_metal_tensor.dart';

class MetalLibrary extends NativeLibrary<MetalBinding> {
  static final instance = MetalLibrary._();

  MetalLibrary._();

  factory MetalLibrary() => instance;

  @override
  String get name => 'Metal';

  @override
  File? tryResolveLibraryFile() {
    return File(
      '/Volumes/safezone/workspace-omnygrid/smollm2/native/macos/metal/build/libsmollm2_metal.dylib',
    );
  }

  @override
  Set<NativeTensorReader> tensorReaders = Set.unmodifiable([
    FP32MetalTensorReader(),
  ]);

  @override
  MetalBinding create() {
    var ptr = metalCreate();
    return MetalBinding._(ptr);
  }

  /////////////////////

  @Native<Pointer<Void> Function()>(symbol: 'metal_create', isLeaf: true)
  external static Pointer<Void> metalCreate();

  @Native<Void Function(Pointer<Void>)>(symbol: 'metal_destroy', isLeaf: true)
  external static void metalDestroy(Pointer<Void> ptr);

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int32, Int32)>(
    symbol: 'metal_set_weights',
    isLeaf: true,
  )
  external static void metalSetWeights(
    Pointer<Void> backend,
    Pointer<Float> weights,
    int rows,
    int cols,
  );

  @Native<
    Void Function(Pointer<Void>, Pointer<Float>, Pointer<Float>, Int, Int)
  >(symbol: 'metal_matmul', isLeaf: true)
  external static void metalMatmul(
    Pointer<Void> backend,
    Pointer<Float> input,
    Pointer<Float> output,
    int rows,
    int cols,
  );

  @Native<
    Void Function(Pointer<Void>, Pointer<Void>, Pointer<Float>, Int, Int)
  >(symbol: 'metal_matmul_input_mltbuffer', isLeaf: true)
  external static void metalMatmulInputMLTBuffer(
    Pointer<Void> backend,
    Pointer<Void> inputBuffer,
    Pointer<Float> output,
    int rows,
    int cols,
  );

  @Native<
    Void Function(Pointer<Void>, Pointer<Float>, Pointer<Void>, Int, Int)
  >(symbol: 'metal_matmul_output_mltbuffer', isLeaf: true)
  external static void metalMatmulOutputMLTBuffer(
    Pointer<Void> backend,
    Pointer<Float> input,
    Pointer<Void> outputBuffer,
    int rows,
    int cols,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Void>, Pointer<Void>, Int, Int)>(
    symbol: 'metal_matmul_input_output_mltbuffer',
    isLeaf: true,
  )
  external static void metalMatmulInputOutputMLTBuffer(
    Pointer<Void> backend,
    Pointer<Void> inputBuffer,
    Pointer<Void> outputBuffer,
    int rows,
    int cols,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int, Int, Int)>(
    symbol: 'metal_copy_weights',
    isLeaf: true,
  )
  external static void metalCopyWeights(
    Pointer<Void> backend,
    Pointer<Float> dst,
    int dstStart,
    int srcStart,
    int length,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int, Int, Int)>(
    symbol: 'metal_copy_output',
    isLeaf: true,
  )
  external static void metalCopyOutput(
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
    symbol: 'metal_create_float_buffer',
    isLeaf: true,
  )
  external static Pointer<Void> metalCreateFloatBuffer(int length);

  @Native<Void Function(Pointer<Void>)>(
    symbol: 'metal_destroy_buffer',
    isLeaf: true,
  )
  external static void metalDestroyBuffer(Pointer<Void> buffer);

  @Native<
    Void Function(
      Pointer<Void>, // MTLBuffer
      Pointer<Float>, // src float array
      Int, // dstStart
      Int, // srcStart
      Int, // count
    )
  >(symbol: 'metal_set_range', isLeaf: true)
  external static void metalSetRange(
    Pointer<Void> buffer,
    Pointer<Float> src,
    int dstStart,
    int srcStart,
    int count,
  );

  @Native<
    Void Function(
      Pointer<Void>, // MTLBuffer
      Pointer<Float>, // dst float array
      Int, // srcStart
      Int, // dstStart
      Int, // count
    )
  >(symbol: 'metal_copy_data_to', isLeaf: true)
  external static void metalCopyDataTo(
    Pointer<Void> buffer,
    Pointer<Float> dst,
    int srcStart,
    int dstStart,
    int count,
  );
}

class MetalBinding implements NativeBinding {
  final Pointer<Void> _ptr;

  MetalBinding._(this._ptr);

  void setWeights(Float32List weights, int rows, int cols) {
    assert(MetalLibrary.instance.isLibraryLoaded);
    MetalLibrary.metalSetWeights(_ptr, weights.address, rows, cols);
  }

  void metalMatmul(Float32List input, Float32List output, int rows, int cols) {
    assert(MetalLibrary.instance.isLibraryLoaded);
    MetalLibrary.metalMatmul(_ptr, input.address, output.address, rows, cols);
  }

  void metalMatmulInputMLTBuffer(
    Pointer<Void> inputBuffer,
    Float32List output,
    int rows,
    int cols,
  ) {
    assert(MetalLibrary.instance.isLibraryLoaded);
    MetalLibrary.metalMatmulInputMLTBuffer(
      _ptr,
      inputBuffer,
      output.address,
      rows,
      cols,
    );
  }

  void metalMatmulOutputMLTBuffer(
    Float32List input,
    Pointer<Void> outputBuffer,
    int rows,
    int cols,
  ) {
    assert(MetalLibrary.instance.isLibraryLoaded);

    MetalLibrary.metalMatmulOutputMLTBuffer(
      _ptr,
      input.address,
      outputBuffer,
      rows,
      cols,
    );
  }

  void metalMatmulInputOutputMLTBuffer(
    Pointer<Void> inputBuffer,
    Pointer<Void> outputBuffer,
    int rows,
    int cols,
  ) {
    assert(MetalLibrary.instance.isLibraryLoaded);

    MetalLibrary.metalMatmulInputOutputMLTBuffer(
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
    assert(MetalLibrary.instance.isLibraryLoaded);

    final length = dstEnd - dstStart;

    assert(dstStart >= 0);
    assert(dstEnd >= dstStart);
    assert(dstEnd <= dst.length);
    assert(dataStart >= 0);

    MetalLibrary.metalCopyWeights(
      _ptr,
      dst.address,
      dstStart,
      dataStart,
      length,
    );
  }

  void copyOutputTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    assert(MetalLibrary.instance.isLibraryLoaded);

    final length = dstEnd - dstStart;

    assert(dstStart >= 0);
    assert(dstEnd >= dstStart);
    assert(dstEnd <= dst.length);
    assert(dataStart >= 0);

    MetalLibrary.metalCopyOutput(
      _ptr,
      dst.address,
      dstStart,
      dataStart,
      length,
    );
  }
}

class MetalFloat32Buffer {
  static final Finalizer<Pointer<Void>> _finalizer = Finalizer<Pointer<Void>>((
    ptr,
  ) {
    // Destroy MLBuffer when GC collects the MetalFloat32Buffer instance.
    MetalLibrary.metalDestroyBuffer(ptr);
  });

  final Pointer<Void> _ptr;
  final int length;

  MetalFloat32Buffer._(this._ptr, this.length) {
    _finalizer.attach(this, _ptr, detach: this);
  }

  factory MetalFloat32Buffer(int length) {
    assert(MetalLibrary.instance.isLibraryLoaded);
    final ptr = MetalLibrary.metalCreateFloatBuffer(length);
    return MetalFloat32Buffer._(ptr, length);
  }

  Pointer<Void> get ptr => _ptr;

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

    MetalLibrary.metalSetRange(_ptr, iterable.address, start, skipCount, count);
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

    MetalLibrary.metalCopyDataTo(_ptr, dst.address, dataStart, dstStart, count);
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

    MetalLibrary.addGpuBufferToCpuBuffer(_ptr, dst.address, length);
  }

  bool _disposed = false;

  bool get isDisposed => _disposed;

  void dispose() {
    if (_disposed) {
      return;
    }

    assert(MetalLibrary.instance.isLibraryLoaded);
    _disposed = true;
    _finalizer.detach(this);
    MetalLibrary.metalDestroyBuffer(_ptr);
  }

  @override
  String toString() => 'MetalFloat32Buffer#${_ptr.address}{length: $length}';
}
