import 'dart:typed_data';

import '../../../../data.dart';
import '../../../../quant_type.dart';
import '../../../tensor.dart';
import '../../native_tensor.dart';
import 'cpu_ffi.dart';

class CPUTensorFloat32Data extends TensorFloat32Data {
  CPUTensorFloat32Data.from(super.data) : super.from();

  CPUTensorFloat32Data(super.length) : super();

  @override
  String toString() {
    return 'CPUTensorFloat32Data#${identityHashCode(this)}@{length: $length}';
  }
}

class FP32CPUTensor extends FP32TensorBase implements NativeTensor {
  static bool? _boot;

  static bool boot({bool tryLoad = false}) {
    if (_boot != null) return _boot!;
    _boot = false;

    if (tryLoad) {
      return _boot = CPULibrary.instance.loadLibrary();
    } else {
      CPULibrary.instance.loadLibraryOrThrow();
      return _boot = true;
    }
  }

  final Float32ListX4 data;

  FP32CPUTensor(super.rows, super.cols, Float32List data)
    : data = data.asFloat32ListX4 {
    if (!colsX4Compatible) {
      throw StateError(
        "`FP32MetalTensor` not compatible with rows:$rows and cols:$cols not X4 compatible!",
      );
    }

    if (!boot()) {
      throw UnsupportedError('`FP32CPUTensor` not supported!');
    }
  }

  @override
  QuantType? get quantType => null;

  factory FP32CPUTensor.vector(Float32List data) {
    return FP32CPUTensor(1, data.length, data);
  }

  factory FP32CPUTensor.readFrom(DataReader dataReader, int size) {
    final raw = dataReader.readBytes(size * 4);

    final bd = ByteData.sublistView(raw);
    final data = Float32List(size);

    for (var i = 0; i < size; i++) {
      data[i] = bd.getFloat32(i * 4, Endian.little);
    }

    return FP32CPUTensor.vector(data);
  }

  @override
  void setRange(
    int start,
    int end,
    Iterable<double> iterable, [
    int skipCount = 0,
  ]) {
    data.list.setRange(start, end, iterable, skipCount);
  }

  @override
  void copyDataTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    dst.setRange(dstStart, dstEnd, data.list, dataStart);
  }

  @override
  FP32Tensor toFP32Tensor({bool cached = true}) =>
      FP32Tensor(rows, cols, data.list);

  @override
  String toString() =>
      'FP32Tensor{size: $size, rows: $rows cols: $cols, data: ${data.list.length}}';

  @override
  void dotTo(TensorFloat32Data out, TensorFloat32Data input) {
    assert(colsX4Compatible);
    _dotToX4(out.array, input);
  }

  late final _cpuBinding = CPULibrary.instance.create();

  void _dotToX4(final Float32List out, final TensorFloat32Data input) {
    _cpuBinding.cpuMatmul(data.list, input.array, out, rows, cols);
  }

  @override
  Float32List get dataArray => data.list;
}

class FP32CPUTensorReader extends NativeTensorReader<FP32CPUTensor> {
  @override
  int get priority => 100;

  @override
  Set<QuantType> get supportedQuantTypes => {QuantType.bf16};

  @override
  FP32CPUTensor readTensor(
    QuantType dataQuantType,
    DataReader dataReader,
    int rows,
    int cols, {
    bool asFP32 = false,
  }) {
    switch (dataQuantType) {
      case QuantType.bf16:
        {
          final bf16Tensor = BF16Tensor.readFromH(dataReader, rows, cols);
          var fp32tensor = bf16Tensor.toFP32Tensor(cached: false);
          return fp32tensor.toFP32CPUTensor();
        }

      default:
        throw UnsupportedError(
          "Can't read from data with `QuantType`: $dataQuantType",
        );
    }
  }
}

extension FP32TensorToFP32CPUTensorExtension on FP32Tensor {
  FP32CPUTensor toFP32CPUTensor() => FP32CPUTensor(rows, cols, dataArray);
}
