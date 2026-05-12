import 'dart:math' as math;
import 'dart:typed_data';

import '../../data.dart';
import '../../quant_type.dart';
import 'dart_tensor.dart';

class Q8Tensor extends QTensor implements DartTensor {
  static const double averageFP32ToQ8Error = 0.03125;
  static const double defaultQ8DequantizationJitterScale = averageFP32ToQ8Error;

  final double scale;

  late final Int8List data;

  Q8Tensor(super.rows, super.cols, this.scale, this.data);

  factory Q8Tensor.readFrom(
    DataReader br,
    int rows,
    int cols, {
    bool readDataHash = true,
  }) {
    final scale = br.readF32();

    (int, int)? hash;
    if (readDataHash) {
      var hash1 = br.readU32();
      var hash2 = br.readU32();
      hash = (hash1, hash2);
    }

    var bytes = br.readBytes(rows * cols);
    var data = bytes.buffer.asInt8List();

    var q8tensor = Q8Tensor(rows, cols, scale, data);

    if (readDataHash) {
      var q8DataHash = q8tensor.data.hashListInt2();
      if (hash != q8DataHash) {
        throw StateError(
          'Q8Tensor integrity check failed while loading tensor ($rows x $cols).\n'
          'Expected hash: $hash\n'
          'Actual hash: $q8DataHash\n'
          'Possible causes: corrupted file, wrong byte length (expected ${rows * cols} bytes), '
          'or mismatched tensor format/version.',
        );
      }
    }

    return q8tensor;
  }

  @override
  QuantType get quantType => QuantType.q8;

  FP32Tensor? _fp32Tensor;

  @override
  FP32Tensor toFP32Tensor({
    bool cached = false,
    math.Random? jitterRandom,
    double jitterScale = defaultQ8DequantizationJitterScale,
  }) {
    if (cached) {
      return _fp32Tensor ??= _toFP32TensorImpl(jitterRandom, jitterScale);
    }
    return _toFP32TensorImpl(jitterRandom, jitterScale);
  }

  FP32Tensor _toFP32TensorImpl(math.Random? jitterRandom, double jitterScale) {
    final scale = this.scale;
    final len = data.length;
    final dataFP32 = Float32List(len);

    if (jitterRandom != null) {
      for (int i = 0; i < len; i++) {
        dataFP32[i] = dequantizeJittered(
          data[i],
          scale,
          jitterRandom,
          jitterScale: jitterScale,
        );
      }
    } else {
      for (int i = 0; i < len; i++) {
        dataFP32[i] = dequantize(data[i], scale);
      }
    }

    return FP32Tensor(rows, cols, dataFP32);
  }

  @override
  String toString() =>
      'Q8Tensor{size: $size, rows: $rows cols: $cols, data: ${data.length}}';

  @override
  void dotTo(TensorFloat32Data out, TensorFloat32Data input) {
    final fp32Tensor = _fp32Tensor;

    if (fp32Tensor != null) {
      fp32Tensor.dotTo(out, input);
    } else if (colsX4Compatible) {
      _dotToX4(out.array, input);
    } else {
      _dotToAny(out.array, input);
    }
  }

  void _dotToX4(final Float32List out, final TensorFloat32Data input) {
    final zeroX4 = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;
    final scale = this.scale;

    final xArrayX4 = input.arrayX4;
    final data = this.data;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      var vsum = zeroX4;

      for (int k = 0; k < simdCols; k++) {
        final p = rowBase + (k << 2);

        final wv = Float32x4(
          data[p].toDouble(),
          data[p + 1].toDouble(),
          data[p + 2].toDouble(),
          data[p + 3].toDouble(),
        );

        vsum += xArrayX4[k] * wv;
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;
      out[i] = sum * scale;
    }
  }

  void _dotToAny(final Float32List out, final TensorFloat32Data input) {
    final zeroX4 = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;
    final scale = this.scale;

    final xArray = input.array;
    final xArrayX4 = input.arrayX4;

    final data = this.data;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      var vsum = zeroX4;

      for (int k = 0; k < simdCols; k++) {
        final p = rowBase + (k << 2);

        final wv = Float32x4(
          data[p].toDouble(),
          data[p + 1].toDouble(),
          data[p + 2].toDouble(),
          data[p + 3].toDouble(),
        );

        vsum += xArrayX4[k] * wv;
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += data[rowBase + j] * xArray[j];
      }

      out[i] = sum * scale;
    }
  }
}
