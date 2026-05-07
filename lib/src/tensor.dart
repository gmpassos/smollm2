import 'dart:math' as math;
import 'dart:typed_data';

import 'data.dart';
import 'quant_type.dart';

abstract class Tensor {
  final int size;
  final int rows;
  final int cols;

  final bool colsX4Compatible;

  Tensor(this.rows, this.cols)
    : colsX4Compatible = cols % 4 == 0,
      size = rows * cols;

  QuantType? get quantType;

  void dotTo(Float32List out, Float32ListX4 x);

  FP32Tensor toFP32Tensor({bool cached = false});
}

abstract class QTensor extends Tensor {
  QTensor(super.rows, super.cols);

  static const double defaultDequantizationJitterScale = 0.008;

  @pragma('vm:prefer-inline')
  double dequantize(int qValue, double scale) {
    // return dequantizeAdaptive(qValue, scale);
    return dequantizeStandard(qValue, scale);
  }

  @pragma('vm:prefer-inline')
  double dequantizeStandard(int qValue, double scale) {
    var v = qValue * scale;
    return v;
  }

  @pragma('vm:prefer-inline')
  double dequantizeCompensated(int qValue, double scale) {
    if (qValue == 0) return 0.0;
    var v = (qValue + (qValue > 0 ? (0.5 / 4) : (-0.5 / 4))) * scale;
    return v;
  }

  @pragma('vm:prefer-inline')
  double dequantizeJittered(
    int qValue,
    double scale,
    math.Random jitterRandom, {

    double jitterScale = QTensor.defaultDequantizationJitterScale,
  }) {
    return dequantizeAdaptiveJittered(
      qValue,
      scale,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
    );
  }

  @pragma('vm:prefer-inline')
  double dequantizeCompensatedStochastic(
    int qValue,
    double scale,
    math.Random jitterRandom, {
    double jitterScale = QTensor.defaultDequantizationJitterScale,
  }) {
    var v = dequantizeCompensated(qValue, scale);

    // deterministic micro stochastic reconstruction
    var j = 1.0 + ((jitterRandom.nextDouble() * 2.0 - 1.0) * jitterScale);
    var v2 = v * j;

    // print('!!! QJ: $v');

    return v2;
  }

  @pragma('vm:prefer-inline')
  double dequantizeAdaptive(
    int qValue,
    double scale, {
    int maxQ = 32767,
    double baseOffset = 0.125,
    double edgeBoost = 0.35,
    double nonLinear = 0.5,
  }) {
    if (qValue == 0) return 0.0;

    final sign = qValue > 0 ? 1.0 : -1.0;
    final absQ = qValue.abs().toDouble();

    // normalized magnitude
    final t = absQ / maxQ;

    // stronger compensation near saturation
    final adaptiveOffset = baseOffset + math.pow(t, nonLinear) * edgeBoost;

    // centroid-style reconstruction
    var reconstructed = (qValue + sign * adaptiveOffset) * scale;

    return reconstructed;
  }

  @pragma('vm:prefer-inline')
  double dequantizeAdaptiveJittered(
    int qValue,
    double scale, {
    int maxQ = 32767,
    double baseOffset = 0.125,
    double edgeBoost = 0.35,
    double nonLinear = 0.5,
    math.Random? jitterRandom,
    double jitterScale = 0.01,
  }) {
    if (qValue == 0) return 0.0;

    final sign = qValue > 0 ? 1.0 : -1.0;
    final absQ = qValue.abs().toDouble();

    // normalized magnitude
    final t = absQ / maxQ;

    // stronger compensation near saturation
    final adaptiveOffset = baseOffset + math.pow(t, nonLinear) * edgeBoost;

    // centroid-style reconstruction
    var reconstructed = (qValue + sign * adaptiveOffset) * scale;

    // optional micro stochastic correction
    if (jitterRandom != null) {
      final jitter =
          (jitterRandom.nextDouble() * 2.0 - 1.0) * jitterScale * scale;

      reconstructed += jitter;
    }

    return reconstructed;
  }

  @override
  FP32Tensor toFP32Tensor({
    bool cached = false,
    math.Random? jitterRandom,
    double jitterScale = defaultDequantizationJitterScale,
  });
}

class Q8Tensor extends QTensor {
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
  void dotTo(Float32List out, Float32ListX4 x) {
    final fp32Tensor = _fp32Tensor;

    if (fp32Tensor != null) {
      fp32Tensor.dotTo(out, x);
    } else if (colsX4Compatible) {
      _dotToX4(out, x);
    } else {
      _dotToAny(out, x);
    }
  }

  void _dotToX4(final Float32List out, final Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;
    final scale = this.scale;

    final xListX4 = x.listX4;
    final data = this.data;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        final p = rowBase + (k << 2);

        final wv = Float32x4(
          data[p].toDouble(),
          data[p + 1].toDouble(),
          data[p + 2].toDouble(),
          data[p + 3].toDouble(),
        );

        vsum += xListX4[k] * wv;
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;
      out[i] = sum * scale;
    }
  }

  void _dotToAny(final Float32List out, final Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;
    final scale = this.scale;

    final xList = x.list;
    final xListX4 = x.listX4;

    final data = this.data;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        final p = rowBase + (k << 2);

        final wv = Float32x4(
          data[p].toDouble(),
          data[p + 1].toDouble(),
          data[p + 2].toDouble(),
          data[p + 3].toDouble(),
        );

        vsum += xListX4[k] * wv;
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += data[rowBase + j] * xList[j];
      }

      out[i] = sum * scale;
    }
  }
}

class Q16Tensor extends QTensor {
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
  void dotTo(Float32List out, Float32ListX4 x) {
    final cached = _fp32Tensor;

    if (cached != null) {
      cached.dotTo(out, x);
    } else if (colsX4Compatible) {
      _dotToX4(out, x);
    } else {
      _dotToAny(out, x);
    }
  }

  void _dotToX4(Float32List out, Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final scale = this.scale;

    final xListX4 = x.listX4;
    final data = this.data;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        final p = rowBase + (k << 2);

        final wv = Float32x4(
          data[p].toDouble(),
          data[p + 1].toDouble(),
          data[p + 2].toDouble(),
          data[p + 3].toDouble(),
        );

        vsum += xListX4[k] * wv;
      }

      final sum = vsum.x + vsum.y + vsum.z + vsum.w;
      out[i] = sum * scale;
    }
  }

  void _dotToAny(Float32List out, Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final scale = this.scale;

    final xList = x.list;
    final xListX4 = x.listX4;

    final data = this.data;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        final p = rowBase + (k << 2);

        final wv = Float32x4(
          data[p].toDouble(),
          data[p + 1].toDouble(),
          data[p + 2].toDouble(),
          data[p + 3].toDouble(),
        );

        vsum += xListX4[k] * wv;
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += data[rowBase + j] * xList[j];
      }

      out[i] = sum * scale;
    }
  }
}

class BF16Tensor extends Tensor {
  final Uint16List data;

  BF16Tensor(super.rows, super.cols, this.data);

  @override
  QuantType? get quantType => QuantType.bf16;

  factory BF16Tensor.vector(Uint16List data) {
    return BF16Tensor(1, data.length, data);
  }

  factory BF16Tensor.readFrom(DataReader dataReader, int rows, int cols) {
    var size = rows * cols;

    final raw = dataReader.readBytes(size * 2);

    final bd = ByteData.sublistView(raw);
    final data = Uint16List(size);

    for (var i = 0; i < size; i++) {
      data[i] = bd.getUint16(i * 2, Endian.little);
    }

    return BF16Tensor(rows, cols, data);
  }

  factory BF16Tensor.readFromH(DataReader dataReader, int rows, int cols) {
    final size = rows * cols;

    final raw = dataReader.readBytes(size * 2);

    final bd = ByteData.sublistView(raw);
    final data = Uint16List(size);

    final hash = Hash64();

    for (var i = 0; i < size; i++) {
      final v = bd.getUint16(i * 2, Endian.little);
      data[i] = v;
      hash.add16(v);
    }

    final expectedH1 = dataReader.readU32();
    final expectedH2 = dataReader.readU32();
    final (h1, h2) = hash.finish();

    if (h1 != expectedH1 || h2 != expectedH2) {
      throw StateError("BF16Tensor hash mismatch");
    }

    return BF16Tensor(rows, cols, data);
  }

  FP32Tensor? _fp32Cache;

  @override
  FP32Tensor toFP32Tensor({bool cached = false}) {
    if (cached) {
      return _fp32Cache ??= _toFP32();
    }
    return _toFP32();
  }

  FP32Tensor _toFP32() {
    final out = Float32List(rows * cols);
    final bd = ByteData(4);

    bd.setUint32(0, 0, Endian.little);

    for (int i = 0; i < data.length; i++) {
      bd.setUint16(2, data[i], Endian.little);
      out[i] = bd.getFloat32(0, Endian.little);
    }

    return FP32Tensor(rows, cols, out);
  }

  // ---- BF16 fast decode ----
  static double _bf16ToF32(int v, ByteData bd) {
    bd.setUint32(0, v << 16, Endian.little);
    return bd.getFloat32(0, Endian.little);
  }

  @override
  void dotTo(Float32List out, Float32ListX4 x) {
    if (colsX4Compatible) {
      _dotToX4(out, x);
    } else {
      _dotToAny(out, x);
    }
  }

  void _dotToX4(Float32List out, Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final xListX4 = x.listX4;

    final data = this.data;

    final bd = ByteData(4);

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        final base = rowBase + (k * 4);

        final a = Float32x4(
          _bf16ToF32(data[base], bd),
          _bf16ToF32(data[base + 1], bd),
          _bf16ToF32(data[base + 2], bd),
          _bf16ToF32(data[base + 3], bd),
        );

        vsum += a * xListX4[k];
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += _bf16ToF32(data[rowBase + j], bd) * x.list[j];
      }

      out[i] = sum;
    }
  }

  void _dotToAny(Float32List out, Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final xList = x.list;
    final xListX4 = x.listX4;

    final data = this.data;

    final bd = ByteData(4);

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        final base = rowBase + (k * 4);

        final a = Float32x4(
          _bf16ToF32(data[base], bd),
          _bf16ToF32(data[base + 1], bd),
          _bf16ToF32(data[base + 2], bd),
          _bf16ToF32(data[base + 3], bd),
        );

        vsum += a * xListX4[k];
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += _bf16ToF32(data[rowBase + j], bd) * xList[j];
      }

      out[i] = sum;
    }
  }
}

class FP32Tensor extends Tensor {
  final Float32ListX4 data;

  FP32Tensor(super.rows, super.cols, Float32List data)
    : data = data.asFloat32ListX4;

  @override
  QuantType? get quantType => null;

  factory FP32Tensor.vector(Float32List data) {
    return FP32Tensor(1, data.length, data);
  }

  factory FP32Tensor.readFrom(DataReader dataReader, int size) {
    final raw = dataReader.readBytes(size * 4);

    final bd = ByteData.sublistView(raw);
    final data = Float32List(size);

    for (var i = 0; i < size; i++) {
      data[i] = bd.getFloat32(i * 4, Endian.little);
    }

    return FP32Tensor.vector(data);
  }

  @override
  FP32Tensor toFP32Tensor({bool cached = true}) => this;

  @override
  String toString() =>
      'FP32Tensor{size: $size, rows: $rows cols: $cols, data: ${data.list.length}}';

  @override
  void dotTo(Float32List out, Float32ListX4 x) {
    if (colsX4Compatible) {
      _dotToX4(out, x);
    } else {
      _dotToAny(out, x);
    }
  }

  void _dotToX4(final Float32List out, final Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final simdCols = cols >> 2;

    final xListX4 = x.listX4;
    final dataX4 = data.listX4;

    for (int i = 0; i < rows; i++) {
      final rowBase = i * simdCols;

      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        vsum += dataX4[rowBase + k] * xListX4[k];
      }

      out[i] = vsum.x + vsum.y + vsum.z + vsum.w;
    }
  }

  void _dotToAny(final Float32List out, final Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final xList = x.list;
    final xListX4 = x.listX4;

    final dataList = data.list;
    final dataX4 = data.listX4;

    for (int i = 0; i < rows; i++) {
      final rowBase4 = i * simdCols;
      final rowBase = i * cols;

      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        vsum += dataX4[rowBase4 + k] * xListX4[k];
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += dataList[rowBase + j] * xList[j];
      }

      out[i] = sum;
    }
  }
}
