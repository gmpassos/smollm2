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

  @Native<
    Void Function(Pointer<Float>, Pointer<Float>, Pointer<Float>, Int32, Int32)
  >(symbol: 'cpu_compute_logits', isLeaf: true)
  external static void cpuComputeLogits(
    Pointer<Float> x,
    Pointer<Float> emb,
    Pointer<Float> logits,
    int hs,
    int vocabSize,
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

  void cpuComputeLogits(
    Float32List x,
    Float32List emb,
    Float32List logits,
    int hs,
    int vocabSize,
  ) => CPULibrary.cpuComputeLogits(
    x.address,
    emb.address,
    logits.address,
    hs,
    vocabSize,
  );
}
