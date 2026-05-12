import 'dart:math' as math;

import '../../data.dart';
import '../../quant_type.dart';
import '../tensor.dart';

export '../tensor.dart';

abstract class DartTensor extends Tensor {
  DartTensor(super.rows, super.cols);
}

class DartTensorReader<T extends DartTensor> extends TensorReader<T> {
  Set<QuantType> get supportedQuantTypes => QuantType.values.toSet();

  @override
  int get priority => 10;

  @override
  Tensor readTensor(
    QuantType dataQuantType,
    DataReader dataReader,
    int rows,
    int cols, {
    bool asFP32 = false,
  }) {
    switch (dataQuantType) {
      case QuantType.q8:
        return _readQ8(dataReader, rows, cols, asFP32: asFP32);

      case QuantType.q16:
        return _readQ16(dataReader, rows, cols, asFP32: asFP32);

      case QuantType.bf16:
        return _readBF16(dataReader, rows, cols, asFP32: asFP32);

      default:
        throw UnsupportedError(
          "Unsupported Dart tensor `QuantType`: ${dataQuantType.name}",
        );
    }
  }

  Tensor _readQ8(
    DataReader br,
    int rows,
    int cols, {
    bool asFP32 = false,
    math.Random? jitterRandom,
    double jitterScale = Q8Tensor.defaultQ8DequantizationJitterScale,
  }) {
    final q8 = Q8Tensor.readFrom(br, rows, cols);

    if (asFP32) {
      return q8.toFP32Tensor(
        cached: false,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
      );
    }

    return q8;
  }

  Tensor _readQ16(
    DataReader br,
    int rows,
    int cols, {
    bool asFP32 = false,
    math.Random? jitterRandom,
    double jitterScale = Q16Tensor.defaultQ16DequantizationJitterScale,
  }) {
    final q16 = Q16Tensor.readFrom(br, rows, cols);

    if (asFP32) {
      return q16.toFP32Tensor(
        cached: false,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
      );
    }

    return q16;
  }

  Tensor _readBF16(DataReader br, int rows, int cols, {bool asFP32 = false}) {
    final bf16 = BF16Tensor.readFromH(br, rows, cols);

    if (asFP32) {
      return bf16.toFP32Tensor(cached: false);
    }

    return bf16;
  }
}
