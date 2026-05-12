import 'dart:typed_data';

import '../../data.dart';
import '../../quant_type.dart';
import 'dart_tensor.dart';

class FP32Tensor extends FP32TensorBase implements DartTensor {
  final TensorFloat32Data data;

  FP32Tensor(super.rows, super.cols, Float32List data)
    : data = TensorFloat32Data.from(data);

  @override
  QuantType? get quantType => null;

  @override
  Float32List get dataArray => data.array;

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
  void setRange(
    int start,
    int end,
    Iterable<double> iterable, [
    int skipCount = 0,
  ]) {
    data.array.setRange(start, end, iterable);
  }

  @override
  void copyDataTo(
    Float32List dst,
    int dstStart,
    int dstEnd, [
    int dataStart = 0,
  ]) {
    dst.setRange(dstStart, dstEnd, data.array, dataStart);
  }

  @override
  FP32Tensor toFP32Tensor({bool cached = true}) => this;

  @override
  String toString() =>
      'FP32Tensor{size: $size, rows: $rows cols: $cols, data: ${data.length}}';

  @override
  void dotTo(TensorFloat32Data out, TensorFloat32Data input) {
    if (colsX4Compatible) {
      _dotToX4(out.array, input);
    } else {
      _dotToAny(out.array, input);
    }
  }

  void _dotToX4(final Float32List out, final TensorFloat32Data input) {
    final rows = this.rows;
    final cols4 = cols ~/ 4;

    final dataArrayX4 = data.arrayX4;
    final xArrayX4 = input.arrayX4;
    final zeroX4 = Float32x4.zero();

    for (int i = 0; i < rows; i++) {
      final rowOffset = i * cols4;

      var sum = zeroX4;

      int a = rowOffset;
      int b = 0;
      final end = rowOffset + cols4;

      // tight loop: best chance for bounds-check elimination
      while (a < end) {
        sum += dataArrayX4[a] * xArrayX4[b];
        a++;
        b++;
      }

      // scalar horizontal sum
      out[i] = sum.x + sum.y + sum.z + sum.w;
    }
  }

  void _dotToAny(final Float32List out, final TensorFloat32Data input) {
    final zeroX4 = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final xArray = input.array;
    final xArrayX4 = input.arrayX4;

    final dataArray = data.array;
    final dataArrayX4 = data.arrayX4;

    for (int i = 0; i < rows; i++) {
      final rowBase4 = i * simdCols;
      final rowBase = i * cols;

      var vsum = zeroX4;

      for (int k = 0; k < simdCols; k++) {
        vsum += dataArrayX4[rowBase4 + k] * xArrayX4[k];
      }

      double sum = vsum.x + vsum.y + vsum.z + vsum.w;

      for (int j = simdCols << 2; j < cols; j++) {
        sum += dataArray[rowBase + j] * xArray[j];
      }

      out[i] = sum;
    }
  }
}
