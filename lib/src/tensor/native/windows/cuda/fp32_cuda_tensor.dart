import 'dart:typed_data';

import '../../../../data.dart';
import '../../../../quant_type.dart';
import '../../../tensor.dart';
import '../../native_tensor.dart';
import 'cuda_ffi.dart';

class CudaTensorFloat32Data implements TensorFloat32Data {
  final CudaFloat32Buffer _data;

  CudaTensorFloat32Data.from(this._data);

  CudaTensorFloat32Data(int length) : _data = CudaFloat32Buffer(length);

  @override
  bool get isInDartMemory => false;

  @override
  Type get dataType => CudaFloat32Buffer;

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
  String toString() => 'CudaTensorFloat32Data@$_data';
}

class FP32CudaTensor extends FP32TensorBase implements NativeTensor {
  static bool? _boot;

  static bool boot({bool tryLoad = false}) {
    if (_boot != null) return _boot!;
    _boot = false;

    if (tryLoad) {
      return _boot = CudaLibrary.instance.loadLibrary();
    } else {
      CudaLibrary.instance.loadLibraryOrThrow();
      return _boot = true;
    }
  }

  final int dataLength;

  late final CudaBinding _cudaBinding;

  FP32CudaTensor(super.rows, super.cols, Float32List data)
    : dataLength = data.length {
    if (!colsX4Compatible) {
      throw StateError(
        "`FP32CudaTensor` not compatible with "
        "rows:$rows and cols:$cols not X4 compatible!",
      );
    }

    if (!boot()) {
      throw UnsupportedError('`FP32CudaTensor` not supported!');
    }

    final cb = _cudaBinding = CudaLibrary.instance.create();

    cb.setWeights(data, rows, cols);
  }

  @override
  QuantType? get quantType => null;

  factory FP32CudaTensor.vector(Float32List data) {
    return FP32CudaTensor(1, data.length, data);
  }

  factory FP32CudaTensor.readFrom(DataReader dataReader, int size) {
    final raw = dataReader.readBytes(size * 4);

    final bd = ByteData.sublistView(raw);

    final data = Float32List(size);

    for (var i = 0; i < size; i++) {
      data[i] = bd.getFloat32(i * 4, Endian.little);
    }

    return FP32CudaTensor.vector(data);
  }

  @override
  FP32CudaTensor toFP32Tensor({bool cached = true}) => this;

  @override
  void dotTo(TensorFloat32Data out, TensorFloat32Data input) {
    assert(colsX4Compatible);

    if (input is CudaTensorFloat32Data) {
      if (out is CudaTensorFloat32Data) {
        _cudaBinding.cudaMatmulInputOutputCudaBuffer(
          input._data.ptr,
          out._data.ptr,
          rows,
          cols,
        );
      } else {
        _cudaBinding.cudaMatmulInputCudaBuffer(
          input._data.ptr,
          out.array,
          rows,
          cols,
        );
      }
    } else if (out is CudaTensorFloat32Data) {
      _cudaBinding.cudaMatmulOutputCudaBuffer(
        input.array,
        out._data.ptr,
        rows,
        cols,
      );
    } else {
      _cudaBinding.cudaMatmul(input.array, out.array, rows, cols);
    }
  }

  @override
  void computeLogits(
    TensorFloat32Data x,
    Float32List logits,
    int hiddenSize,
    int vocabSize,
  ) {
    _cudaBinding.cudaComputeLogits(x.array, logits, vocabSize, hiddenSize);
  }

  @override
  Float32List get dataArray {
    final out = Float32List(dataLength);
    _cudaBinding.copyWeightsTo(out, 0, dataLength, 0);
    return out;
  }

  @override
  void copyDataTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    _cudaBinding.copyWeightsTo(dst, dstStart, dstEnd, dataStart);
  }

  @override
  String toString() =>
      'FP32CudaTensor{'
      'size: $size, '
      'rows: $rows '
      'cols: $cols, '
      'data: $dataLength'
      '}';
}

class FP32CudaTensorReader extends NativeTensorReader<FP32CudaTensor> {
  @override
  int get priority => 1000;

  @override
  Set<QuantType> get supportedQuantTypes => {QuantType.bf16};

  @override
  FP32CudaTensor readTensor(
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
          '`QuantType`: $dataQuantType',
        );
    }
  }

  FP32CudaTensor _readBF16(DataReader dataReader, int rows, int cols) {
    final bf16Tensor = BF16Tensor.readFromH(dataReader, rows, cols);
    final fp32tensor = bf16Tensor.toFP32Tensor(cached: false);
    return fp32tensor.toFP32CudaTensor();
  }
}

extension FP32TensorToFP32CudaTensorExtension on FP32Tensor {
  FP32CudaTensor toFP32CudaTensor() => FP32CudaTensor(rows, cols, dataArray);
}
