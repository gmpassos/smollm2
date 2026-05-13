// Experimental Q16 with scales per block of 32:
/*
class Q16PerBlockTensor extends QTensor {
  static const int defaultBlockSize = 32;

  final int blockSize;

  /// One scale per block
  final List<double> scales;

  /// Quantized data per block
  final List<Int16List> dataBlocks;

  Q16PerBlockTensor(
    super.rows,
    super.cols,
    this.blockSize,
    this.scales,
    this.dataBlocks,
  ) : assert(scales.length == dataBlocks.length);

  int get blockCount => dataBlocks.length;

  factory Q16PerBlockTensor.readFrom(
    DataReader br,
    int rows,
    int cols, {
    int blockSize = defaultBlockSize,
    bool readDataHash = true,
  }) {
    var blkSz = br.readU16();
    if (blkSz != blockSize) {
      throw StateError("Incompatible block size: $blkSz != $blockSize");
    }

    final size = rows * cols;
    final blockCount = (size + blockSize - 1) ~/ blockSize;

    final scales = List<double>.filled(blockCount, 0.0);

    final dummyInt16List = Int16List(0);
    final dataBlocks = List<Int16List>.filled(blockCount, dummyInt16List);

    final dataFull = Int16List(size);

    for (int b = 0; b < blockCount; b++) {
      final remaining = size - (b * blockSize);
      final blockLength = remaining < blockSize ? remaining : blockSize;

      final scale = br.readF32();

      (int, int)? blockHash;
      if (readDataHash) {
        var hash1 = br.readU32();
        var hash2 = br.readU32();
        blockHash = (hash1, hash2);
      }

      final bytes = br.readBytes(blockLength * 2);
      final bd = ByteData.sublistView(bytes);

      var dataFullOffset = blockSize * b;
      final data = Int16List.sublistView(
        dataFull,
        dataFullOffset,
        dataFullOffset + blockLength,
      );

      for (int i = 0; i < blockLength; i++) {
        data[i] = bd.getInt16(i * 2, Endian.little);
      }

      scales[b] = scale;
      dataBlocks[b] = data;

      if (readDataHash) {
        var q16DataHash = data.hashListInt2();
        if (blockHash != q16DataHash) {
          throw StateError(
            'Q16PerBlockTensor integrity check failed while loading tensor ($rows x $cols) block[$b].\n'
            'Expected hash: $blockHash\n'
            'Actual hash: $q16DataHash\n'
            'Possible causes: corrupted file, wrong byte length (expected ${rows * cols} bytes), '
            'or mismatched tensor format/version.',
          );
        }
      }
    }

    final q16PerBlockTensor = Q16PerBlockTensor(
      rows,
      cols,
      blockSize,
      scales,
      dataBlocks,
    );

    return q16PerBlockTensor;
  }

  @override
  QuantType get quantType => QuantType.q16PerBlock;

  FP32Tensor? _fp32Cache;

  @override
  FP32Tensor toFP32Tensor({
    bool cached = false,
    math.Random? jitterRandom,
    double jitterScale = Q16Tensor.defaultQ16DequantizationJitterScale,
  }) {
    if (cached) {
      return _fp32Cache ??= _toFP32(jitterRandom, jitterScale);
    }
    return _toFP32(jitterRandom, jitterScale);
  }

  FP32Tensor _toFP32(math.Random? jitterRandom, double jitterScale) {
    final out = Float32List(size);

    int offset = 0;

    if (jitterRandom != null) {
      for (int b = 0; b < blockCount; b++) {
        final block = dataBlocks[b];
        final scale = scales[b];

        for (int i = 0; i < block.length; i++) {
          final v = block[i];
          out[offset++] = dequantizeJittered(
            v,
            scale,
            jitterRandom,
            jitterScale: jitterScale,
          );
        }
      }
    } else {
      for (int b = 0; b < blockCount; b++) {
        final block = dataBlocks[b];
        final scale = scales[b];

        for (int i = 0; i < block.length; i++) {
          final v = block[i];
          out[offset++] = dequantize(v, scale);
        }
      }
    }

    return FP32Tensor(rows, cols, out);
  }

  @override
  void dotTo(Float32List out, Float32ListX4 x) {
    if (colsX4Compatible) {
      _dotToX4(out, x);
    } else {
      _dotAny(out, x);
    }
  }

  void _dotToX4(Float32List out, Float32ListX4 x) {
    final zero = Float32x4.zero();

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    final scales = this.scales;

    final xListX4 = x.listX4;
    final dataBlocks = this.dataBlocks;

    var blockI = 0;
    var blockLocalI = 0;
    var block = dataBlocks[blockI];
    var scale = scales[blockI];

    loop:
    for (int i = 0; i < rows; i++) {
      final rowBase = i * cols;

      double scaledSum = 0.0;
      Float32x4 vsum = zero;

      for (int k = 0; k < simdCols; k++) {
        final p = rowBase + (k * 4);

        final wv = Float32x4(
          block[blockLocalI++].toDouble(),
          block[blockLocalI++].toDouble(),
          block[blockLocalI++].toDouble(),
          block[blockLocalI++].toDouble(),
        );

        vsum += xListX4[k] * wv;

        if (blockLocalI == block.length) {
          final sum = vsum.x + vsum.y + vsum.z + vsum.w;
          scaledSum += sum * scale;
          vsum = zero;

          blockI++;
          blockLocalI = 0;

          if (blockI < dataBlocks.length) {
            block = dataBlocks[blockI];
            scale = scales[blockI];
          } else {
            break loop;
          }
        }
      }

      if (blockLocalI > 0) {
        final sum = vsum.x + vsum.y + vsum.z + vsum.w;
        scaledSum += sum * scale;
      }

      out[i] = scaledSum;
    }
  }

  void _dotX4b(Float32List out, Float32ListX4 x) {
    final dataBlocks = this.dataBlocks;
    final scales = this.scales;
    final blockSize = this.blockSize;

    final rows = this.rows;
    final cols = this.cols;
    final simdCols = cols >> 2;

    int globalIndex = 0;

    for (int r = 0; r < rows; r++) {
      Float32x4 acc = Float32x4.zero();

      for (int k = 0; k < simdCols; k++) {
        final x4 = x.listX4[k];

        // process 4 consecutive scalars explicitly but safely
        final i0 = globalIndex;
        final i1 = i0 + 1;
        final i2 = i0 + 2;
        final i3 = i0 + 3;

        final b0 = i0 ~/ blockSize;
        final b1 = i1 ~/ blockSize;
        final b2 = i2 ~/ blockSize;
        final b3 = i3 ~/ blockSize;

        final v = Float32x4(
          dataBlocks[b0][i0 % blockSize].toDouble() * scales[b0],
          dataBlocks[b1][i1 % blockSize].toDouble() * scales[b1],
          dataBlocks[b2][i2 % blockSize].toDouble() * scales[b2],
          dataBlocks[b3][i3 % blockSize].toDouble() * scales[b3],
        );

        acc += x4 * v;

        globalIndex += 4;
      }

      double sum = acc.x + acc.y + acc.z + acc.w;

      for (int c = simdCols << 2; c < cols; c++) {
        final b = globalIndex ~/ blockSize;
        final i = globalIndex % blockSize;

        sum += dataBlocks[b][i].toDouble() * scales[b] * x.list[c];
        globalIndex++;
      }

      out[r] = sum;
    }
  }

  void _dotAny(Float32List out, Float32ListX4 x) {
    final rows = this.rows;
    final cols = this.cols;

    final xList = x.list;
    final xX4 = x.listX4;

    int globalIndex = 0;

    for (int r = 0; r < rows; r++) {
      double sum = 0.0;

      for (int c = 0; c < cols; c++) {
        final b = globalIndex ~/ blockSize;
        final i = globalIndex % blockSize;

        final w = dataBlocks[b][i] * scales[b];
        sum += w * xList[c];

        globalIndex++;
      }

      out[r] = sum;
    }
  }
}
*/
