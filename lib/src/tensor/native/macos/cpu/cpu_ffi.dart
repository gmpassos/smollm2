import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import '../../native_tensor.dart';
import 'fp32_cpu_tensor.dart';

class CPULibrary extends NativeLibrary<CPUBinding> {
  static final instance = CPULibrary._();

  CPULibrary._();

  factory CPULibrary() => instance;

  @override
  String get name => 'CPU';

  @override
  File? tryResolveLibraryFile() {
    return File(
      '/Volumes/safezone/workspace-omnygrid/smollm2/native/macos/cpu/build/libsmollm2_cpu.dylib',
    );
  }

  @override
  Set<NativeTensorReader> tensorReaders = Set.unmodifiable([
    FP32CPUTensorReader(),
  ]);

  @override
  CPUBinding create() {
    return CPUBinding();
  }

  @Native<
    Void Function(Pointer<Float>, Pointer<Float>, Pointer<Float>, Int32, Int32)
  >(symbol: 'cpu_matmul', isLeaf: true)
  external static void cpuMatmul(
    Pointer<Float> weights,
    Pointer<Float> input,
    Pointer<Float> output,
    int rows,
    int cols,
  );
}

class CPUBinding implements NativeBinding {
  CPUBinding();

  void cpuMatmul(
    Float32List data,
    Float32List input,
    Float32List output,
    int rows,
    int cols,
  ) => CPULibrary.cpuMatmul(
    data.address,
    input.address,
    output.address,
    rows,
    cols,
  );
}
