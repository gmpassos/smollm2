import 'dart:math' as math;
import 'dart:typed_data';

import '../../data.dart';
import '../../quant_type.dart';
import 'dart_tensor.dart';

class Q16Tensor extends QTensor implements DartTensor {
  static const double averageFP32ToQ16Error = 0.0009765625;
  static const double defaultQ16DequantizationJitterScale =
      averageFP32ToQ16Error;

  final double scale;

  late final Int16List data;

  Q16Tensor(super.rows, super.cols, this.scale, this.data);

  factory Q16Tensor.readFrom(
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

    final bytes = br.readBytes(rows * cols * 2);
    final bd = ByteData.view(bytes.buffer);

    final data = Int16List(rows * cols);
    for (int i = 0; i < data.length; i++) {
      data[i] = bd.getInt16(i * 2, Endian.little);
    }

    var q16tensor = Q16Tensor(rows, cols, scale, data);

    if (readDataHash) {
      var q16DataHash = q16tensor.data.hashListInt2();
      if (hash != q16DataHash) {
        throw StateError(
          'Q16Tensor integrity check failed while loading tensor ($rows x $cols).\n'
          'Expected hash: $hash\n'
          'Actual hash: $q16DataHash\n'
          'Possible causes: corrupted file, wrong byte length (expected ${rows * cols} bytes), '
          'or mismatched tensor format/version.',
        );
      }
    }

    return q16tensor;
  }

  @override
  QuantType get quantType => QuantType.q16;

  FP32Tensor? _fp32Tensor;

  @override
  FP32Tensor toFP32Tensor({
    bool cached = false,
    math.Random? jitterRandom,
    double jitterScale = defaultQ16DequantizationJitterScale,
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
      'Q16Tensor{size: $size, rows: $rows cols: $cols, data: ${data.length}}';

  @override
  void dotTo(TensorFloat32Data out, TensorFloat32Data input) {
    final cached = _fp32Tensor;

    if (cached != null) {
      cached.dotTo(out, input);
    } else if (colsX4Compatible) {
      _dotToX4(out.array, input);
    } else {
      _dotToAny(out.array, input);
    }
  }

  void _dotToX4(Float32List out, TensorFloat32Data input) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final scale = this.scale;

    final xArrayX4 = input.arrayX4;
    final data = this.data;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      var vsum = zero;

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

      final sum = vsum.x + vsum.y + vsum.z + vsum.w;
      out[i] = sum * scale;
    }
  }

  void _dotToAny(Float32List out, TensorFloat32Data input) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final scale = this.scale;

    final xArray = input.array;
    final xArrayX4 = input.arrayX4;

    final data = this.data;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      var vsum = zero;

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
