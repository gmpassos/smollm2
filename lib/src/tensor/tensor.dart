import 'dart:math' as math;
import 'dart:typed_data';

import '../data.dart';
import '../quant_type.dart';
import 'dart/dart_tensor.dart';
import 'native/native_tensor.dart';

export 'dart/bf16_tensor.dart';
export 'dart/fp32_tensor.dart';
export 'dart/q16_tensor.dart';
export 'dart/q8_tensor.dart';

typedef Float32ListX4 = ({Float32List list, Float32x4List listX4});

extension Float32ListX4Extension on Float32ListX4 {
  int get length => list.length;
}

extension Float32ListExtension on Float32List {
  Float32x4List get asFloat32x4List => buffer.asFloat32x4List();

  Float32ListX4 get asFloat32ListX4 => (list: this, listX4: asFloat32x4List);

  TensorFloat32Data get asTensorFloat32Data => TensorFloat32Data.from(this);
}

typedef TensorFloat32DataCreator = TensorFloat32Data Function(int length);

class TensorFloat32Data {
  final Float32List _data;

  final Float32x4List _dataX4;

  TensorFloat32Data.from(this._data) : _dataX4 = _data.asFloat32x4List;

  TensorFloat32Data(int length) : this.from(Float32List(length));

  bool get isInDartMemory => true;

  Type get dataType => Float32List;

  int get length => _data.length;

  Float32List get array => _data;

  Float32x4List get arrayX4 => _dataX4;

  Float32List arrayView({Float32List? buffer}) => _data;

  void setRange(int start, int end, Float32List iterable, [int skipCount = 0]) {
    _data.setRange(start, end, iterable, skipCount);
  }

  void copyDataTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    dst.setRange(dstStart, dstEnd, _data, dataStart);
  }

  void addTo(Float32List dst, int length) {
    if (length > this.length) {
      throw RangeError.range(
        length,
        0,
        this.length,
        'length',
        'Must be ≤ ${this.length}',
      );
    }

    final src = _data;

    for (int i = 0; i < length; i++) {
      dst[i] += src[i];
    }
  }

  @override
  String toString() =>
      'TensorFloat32Data#${identityHashCode(this)}{length: $length}';
}

class TensorFactory {
  static final instance = TensorFactory._();

  TensorFactory._();

  late final NativeTensorFactory _nativeTensorFactory;

  void load() {
    _nativeTensorFactory = NativeTensorFactory.instance..load();
  }

  static final _dartTensorReader = DartTensorReader();

  TensorReader<Tensor>? getTensorReader(QuantType dataQuantType) {
    var nativeTensorReader = _nativeTensorFactory.getTensorReader(
      dataQuantType,
    );
    if (nativeTensorReader != null) return nativeTensorReader;

    return _dartTensorReader;
  }

  Tensor readTensor(
    QuantType dataQuantType,
    DataReader dataReader,
    int rows,
    int cols, {
    bool asFP32 = false,
  }) {
    var tensorReader =
        getTensorReader(dataQuantType) ??
        (throw UnsupportedError(
          "Unsupported tensor `QuantType`: ${dataQuantType.name}",
        ));

    return tensorReader.readTensor(
      dataQuantType,
      dataReader,
      rows,
      cols,
      asFP32: asFP32,
    );
  }
}

abstract class TensorReader<T extends Tensor>
    implements Comparable<TensorReader> {
  int get priority;

  Tensor readTensor(
    QuantType dataQuantType,
    DataReader dataReader,
    int rows,
    int cols, {
    bool asFP32 = false,
  });

  @override
  int compareTo(TensorReader<Tensor> other) {
    return other.priority.compareTo(priority);
  }
}

abstract class Tensor {
  final int size;
  final int rows;
  final int cols;

  final bool colsX4Compatible;

  Tensor(this.rows, this.cols)
    : colsX4Compatible = cols % 4 == 0,
      size = rows * cols;

  QuantType? get quantType;

  void setRange(
    int start,
    int end,
    Iterable<double> iterable, [
    int skipCount = 0,
  ]) {
    throw UnsupportedError("`setRange` not supported for $runtimeType");
  }

  void copyDataTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    throw UnsupportedError("`copyTo` not supported for $runtimeType");
  }

  void dotTo(TensorFloat32Data out, TensorFloat32Data input);

  FP32TensorBase toFP32Tensor({bool cached = false});
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
  FP32TensorBase toFP32Tensor({
    bool cached = false,
    math.Random? jitterRandom,
    double jitterScale = defaultDequantizationJitterScale,
  });
}

abstract class FP32TensorBase extends Tensor {
  FP32TensorBase(super.rows, super.cols);

  @override
  QuantType? get quantType => null;

  @override
  FP32TensorBase toFP32Tensor({bool cached = true}) => this;

  Float32List get dataArray;
}
