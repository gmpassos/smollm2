import 'dart:typed_data';

import '../../../../data.dart';
import '../../../../quant_type.dart';
import '../../../tensor.dart';
import '../../native_tensor.dart';
import 'metal_ffi.dart';

class MetalTensorFloat32Data implements TensorFloat32Data {
  final MetalFloat32Buffer _data;

  MetalTensorFloat32Data.from(this._data);

  MetalTensorFloat32Data(int length) : _data = MetalFloat32Buffer(length);

  @override
  bool get isInDartMemory => false;

  @override
  Type get dataType => MetalFloat32Buffer;

  @override
  int get length => _data.length;

  @override
  Float32List get array => throw UnimplementedError();

  @override
  Float32x4List get arrayX4 => throw UnimplementedError();

  @override
  Float32List arrayView({Float32List? buffer}) {
    final length = this.length;

    if (buffer == null) {
      buffer = Float32List(length);
    } else if (buffer.length < length) {
      throw ArgumentError.value(
        buffer.length,
        'buffer.length',
        'Buffer too small: ${buffer.length} < required length $length',
      );
    }

    copyDataTo(buffer, 0, length);
    return buffer;
  }

  @override
  void setRange(int start, int end, Float32List iterable, [int skipCount = 0]) {
    _data.setRange(start, end, iterable, skipCount);
  }

  @override
  void copyDataTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    _data.copyDataTo(dst, dstStart, dstEnd, dataStart);
  }

  @override
  void addTo(Float32List dst, int length) {
    _data.addTo(dst, length);
  }

  @override
  String toString() => 'MetalTensorFloat32Data@$_data';
}

class FP32MetalTensor extends FP32TensorBase implements NativeTensor {
  static bool? _boot;

  static bool boot({bool tryLoad = false}) {
    if (_boot != null) return _boot!;
    _boot = false;

    if (tryLoad) {
      return _boot = MetalLibrary.instance.loadLibrary();
    } else {
      MetalLibrary.instance.loadLibraryOrThrow();
      return _boot = true;
    }
  }

  final int dataLength;

  late final MetalBinding _metalBinding;

  FP32MetalTensor(super.rows, super.cols, Float32List data)
    : dataLength = data.length {
    if (!colsX4Compatible) {
      throw StateError(
        "`FP32MetalTensor` not compatible with "
        "rows:$rows and cols:$cols not X4 compatible!",
      );
    }

    if (!boot()) {
      throw UnsupportedError('`FP32MetalTensor` not supported!');
    }

    final mb = _metalBinding = MetalLibrary.instance.create();

    mb.setWeights(data, rows, cols);
  }

  @override
  QuantType? get quantType => null;

  factory FP32MetalTensor.vector(Float32List data) {
    return FP32MetalTensor(1, data.length, data);
  }

  factory FP32MetalTensor.readFrom(DataReader dataReader, int size) {
    final raw = dataReader.readBytes(size * 4);

    final bd = ByteData.sublistView(raw);
    final data = Float32List(size);

    for (var i = 0; i < size; i++) {
      data[i] = bd.getFloat32(i * 4, Endian.little);
    }

    return FP32MetalTensor.vector(data);
  }

  @override
  FP32MetalTensor toFP32Tensor({bool cached = true}) => this;

  @override
  void dotTo(TensorFloat32Data out, TensorFloat32Data input) {
    assert(colsX4Compatible);

    if (input is MetalTensorFloat32Data) {
      if (out is MetalTensorFloat32Data) {
        _metalBinding.metalMatmulInputOutputMLTBuffer(
          input._data.ptr,
          out._data.ptr,
          rows,
          cols,
        );
      } else {
        _metalBinding.metalMatmulInputMLTBuffer(
          input._data.ptr,
          out.array,
          rows,
          cols,
        );
      }
    } else if (out is MetalTensorFloat32Data) {
      _metalBinding.metalMatmulOutputMLTBuffer(
        input.array,
        out._data.ptr,
        rows,
        cols,
      );
    } else {
      _metalBinding.metalMatmul(input.array, out.array, rows, cols);
    }
  }

  @override
  Float32List get dataArray {
    final out = Float32List(dataLength);
    _metalBinding.copyWeightsTo(out, 0, dataLength, 0);
    return out;
  }

  @override
  void copyDataTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    _metalBinding.copyWeightsTo(dst, dstStart, dstEnd, dataStart);
  }

  @override
  String toString() =>
      'FP32Tensor{'
      'size: $size, '
      'rows: $rows '
      'cols: $cols, '
      'data: $dataLength'
      '}';
}

class FP32MetalTensorReader extends NativeTensorReader<FP32MetalTensor> {
  @override
  int get priority => 1000;

  @override
  Set<QuantType> get supportedQuantTypes => {QuantType.bf16};

  @override
  FP32MetalTensor readTensor(
    QuantType dataQuantType,
    DataReader dataReader,
    int rows,
    int cols, {
    bool asFP32 = false,
  }) {
    switch (dataQuantType) {
      case QuantType.bf16:
        return _readBF16(dataReader, rows, cols);

      default:
        throw UnsupportedError(
          "Can't read from data with "
          "`QuantType`: $dataQuantType",
        );
    }
  }

  FP32MetalTensor _readBF16(DataReader dataReader, int rows, int cols) {
    final bf16Tensor = BF16Tensor.readFromH(dataReader, rows, cols);
    var fp32tensor = bf16Tensor.toFP32Tensor(cached: false);
    return fp32tensor.toFP32MetalTensor();
  }
}

extension FP32TensorToFP32MetalTensorExtension on FP32Tensor {
  FP32MetalTensor toFP32MetalTensor() => FP32MetalTensor(rows, cols, dataArray);
}
