import 'dart:typed_data';

import '../../data.dart';
import '../../quant_type.dart';

import 'dart_tensor.dart';

class BF16Tensor extends DartTensor {
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
  void dotTo(TensorFloat32Data out, TensorFloat32Data input) {
    if (colsX4Compatible) {
      _dotToX4(out.array, input);
    } else {
      _dotToAny(out.array, input);
    }
  }

  void _dotToX4(Float32List out, TensorFloat32Data input) {
    final zeroX4 = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final xArray = input.array;
    final xArrayX4 = input.arrayX4;

    final data = this.data;

    final bd = ByteData(4);

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      var vsum = zeroX4;

      for (int k = 0; k < simdCols; k++) {
        final base = rowBase + (k * 4);

        final a = Float32x4(
          _bf16ToF32(data[base], bd),
          _bf16ToF32(data[base + 1], bd),
          _bf16ToF32(data[base + 2], bd),
          _bf16ToF32(data[base + 3], bd),
        );

        vsum += a * xArrayX4[k];
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += _bf16ToF32(data[rowBase + j], bd) * xArray[j];
      }

      out[i] = sum;
    }
  }

  void _dotToAny(Float32List out, TensorFloat32Data input) {
    final zeroX4 = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final xArray = input.array;
    final xArrayX4 = input.arrayX4;

    final data = this.data;

    final bd = ByteData(4);

    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      var vsum = zeroX4;

      for (int k = 0; k < simdCols; k++) {
        final base = rowBase + (k * 4);

        final a = Float32x4(
          _bf16ToF32(data[base], bd),
          _bf16ToF32(data[base + 1], bd),
          _bf16ToF32(data[base + 2], bd),
          _bf16ToF32(data[base + 3], bd),
        );

        vsum += a * xArrayX4[k];
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += _bf16ToF32(data[rowBase + j], bd) * xArray[j];
      }

      out[i] = sum;
    }
  }
}
