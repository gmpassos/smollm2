import 'dart:math' as math;

import 'package:smollm2/src/quant_type.dart';

import 'config.dart';
import 'data.dart';
import 'tensor.dart';

class LayerWeights {
  late FP32Tensor inputLayerNorm;
  late Tensor qProj;
  late Tensor kProj;
  late Tensor vProj;
  late Tensor oProj;

  late FP32Tensor postAttentionLayerNorm;

  late Tensor gateProj;
  late Tensor upProj;
  late Tensor downProj;
}

class ModelWeights {
  final Config config;

  ModelWeights(this.config);

  late Tensor embedTokens;
  late List<LayerWeights> layers;
  late FP32Tensor finalNorm;

  void load(
    Config config,
    DataReader dataReader, {
    math.Random? jitterRandom,
    double? jitterScale,
  }) {
    _loadEmbedTokens(config, dataReader, jitterRandom, jitterScale);
    _loadLayers(config, dataReader, jitterRandom, jitterScale);
    _loadFinalNorm(config, dataReader);
  }

  void _loadEmbedTokens(
    Config config,
    DataReader dataReader,
    math.Random? jitterRandom,
    double? jitterScale,
  ) {
    embedTokens = _readTensor(
      dataReader,
      config.vocabSize,
      config.hiddenSize,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
      quantType: config.quantType,
      fp32: true,
    );
  }

  void _loadLayers(
    Config config,
    DataReader dataReader,
    math.Random? jitterRandom,
    double? jitterScale,
  ) {
    final quantType = config.quantType;
    final hd = config.headDim;

    layers = List.generate(config.numLayers, (_) {
      final lw = LayerWeights();

      ////

      lw.inputLayerNorm = FP32Tensor.readFrom(dataReader, config.hiddenSize);

      ////

      lw.qProj = _readTensor(
        dataReader,
        config.numHeads * hd,
        config.hiddenSize,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
        quantType: quantType,
        fp32: true,
      );

      lw.kProj = _readTensor(
        dataReader,
        config.numKvHeads * hd,
        config.hiddenSize,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
        quantType: quantType,
        fp32: true,
      );

      lw.vProj = _readTensor(
        dataReader,
        config.numKvHeads * hd,
        config.hiddenSize,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
        quantType: quantType,
        fp32: true,
      );

      lw.oProj = _readTensor(
        dataReader,
        config.hiddenSize,
        config.numHeads * hd,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
        quantType: quantType,
        fp32: true,
      );

      ////

      lw.postAttentionLayerNorm = FP32Tensor.readFrom(
        dataReader,
        config.hiddenSize,
      );

      ////

      lw.gateProj = _readTensor(
        dataReader,
        config.intermediateSize,
        config.hiddenSize,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
        quantType: quantType,
        fp32: true,
      );

      lw.upProj = _readTensor(
        dataReader,
        config.intermediateSize,
        config.hiddenSize,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
        quantType: quantType,
        fp32: true,
      );

      lw.downProj = _readTensor(
        dataReader,
        config.hiddenSize,
        config.intermediateSize,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
        quantType: quantType,
        fp32: true,
      );

      return lw;
    });
  }

  void _loadFinalNorm(Config config, DataReader dataReader) {
    finalNorm = FP32Tensor.readFrom(dataReader, config.hiddenSize);
  }

  Tensor _readTensor(
    DataReader br,
    int rows,
    int cols, {
    required QuantType quantType,
    bool fp32 = false,
    math.Random? jitterRandom,
    double? jitterScale,
  }) {
    switch (quantType) {
      case QuantType.q8:
        return _readQ8(
          br,
          rows,
          cols,
          jitterRandom: jitterRandom,
          jitterScale:
              jitterScale ?? Q8Tensor.defaultQ8DequantizationJitterScale,
          fp32: fp32,
        );
      case QuantType.q16:
        return _readQ16(
          br,
          rows,
          cols,
          jitterRandom: jitterRandom,
          jitterScale:
              jitterScale ?? Q16Tensor.defaultQ16DequantizationJitterScale,
          fp32: fp32,
        );
      case QuantType.bf16:
        return _readBF16(
          br,
          rows,
          cols,
          jitterRandom: jitterRandom,
          jitterScale:
              jitterScale ?? Q16Tensor.defaultQ16DequantizationJitterScale,
          fp32: fp32,
        );
      default:
        throw UnsupportedError("quantType: $quantType");
    }
  }

  Tensor _readQ8(
    DataReader br,
    int rows,
    int cols, {
    bool fp32 = false,
    math.Random? jitterRandom,
    double jitterScale = Q8Tensor.defaultQ8DequantizationJitterScale,
  }) {
    final q8 = Q8Tensor.readFrom(br, rows, cols);

    if (fp32) {
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
    bool fp32 = false,
    math.Random? jitterRandom,
    double jitterScale = Q16Tensor.defaultQ16DequantizationJitterScale,
  }) {
    final q16 = Q16Tensor.readFrom(br, rows, cols);
    //final q16 = Q16PerBlockTensor.readFrom(br, rows, cols);

    if (fp32) {
      return q16.toFP32Tensor(
        cached: false,
        jitterRandom: jitterRandom,
        jitterScale: jitterScale,
      );
    }

    return q16;
  }

  Tensor _readBF16(
    DataReader br,
    int rows,
    int cols, {
    bool fp32 = false,
    math.Random? jitterRandom,
    double jitterScale = Q16Tensor.defaultQ16DequantizationJitterScale,
  }) {
    final bf16 = BF16Tensor.readFromH(br, rows, cols);

    if (fp32) {
      return bf16.toFP32Tensor(cached: false);
    }

    return bf16;
  }

  @override
  String toString() =>
      'ModelWeights{embedTokens: $embedTokens, layers: ${layers.length}, finalNorm: $finalNorm}';
}
