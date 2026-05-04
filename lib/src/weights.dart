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

  void load(Config config, DataReader dataReader) {
    _loadEmbedTokens(config, dataReader);
    _loadLayers(config, dataReader);
    _loadFinalNorm(config, dataReader);
  }

  void _loadEmbedTokens(Config config, DataReader dataReader) {
    embedTokens = _readQ(
      dataReader,
      config.vocabSize,
      config.hiddenSize,
      quantType: config.quantType,
      fp32: true,
    );
  }

  void _loadLayers(Config config, DataReader dataReader) {
    final quantType = config.quantType;
    final hd = config.headDim;

    layers = List.generate(config.numLayers, (_) {
      final lw = LayerWeights();

      ////

      lw.inputLayerNorm = FP32Tensor.readFrom(dataReader, config.hiddenSize);

      ////

      lw.qProj = _readQ(
        dataReader,
        config.numHeads * hd,
        config.hiddenSize,
        quantType: quantType,
        fp32: true,
      );

      lw.kProj = _readQ(
        dataReader,
        config.numKvHeads * hd,
        config.hiddenSize,
        quantType: quantType,
        fp32: true,
      );

      lw.vProj = _readQ(
        dataReader,
        config.numKvHeads * hd,
        config.hiddenSize,
        quantType: quantType,
        fp32: true,
      );

      lw.oProj = _readQ(
        dataReader,
        config.hiddenSize,
        config.numHeads * hd,
        quantType: quantType,
        fp32: true,
      );

      ////

      lw.postAttentionLayerNorm = FP32Tensor.readFrom(
        dataReader,
        config.hiddenSize,
      );

      ////

      lw.gateProj = _readQ(
        dataReader,
        config.intermediateSize,
        config.hiddenSize,
        quantType: quantType,
        fp32: true,
      );

      lw.upProj = _readQ(
        dataReader,
        config.intermediateSize,
        config.hiddenSize,
        quantType: quantType,
        fp32: true,
      );

      lw.downProj = _readQ(
        dataReader,
        config.hiddenSize,
        config.intermediateSize,
        quantType: quantType,
        fp32: true,
      );

      return lw;
    });
  }

  void _loadFinalNorm(Config config, DataReader dataReader) {
    finalNorm = FP32Tensor.readFrom(dataReader, config.hiddenSize);
  }

  Tensor _readQ(
    DataReader br,
    int rows,
    int cols, {
    required QuantType quantType,
    bool fp32 = false,
  }) {
    switch (quantType) {
      case QuantType.q8:
        return _readQ8(br, rows, cols, fp32: fp32);
      case QuantType.q16:
        return _readQ16(br, rows, cols, fp32: fp32);
      default:
        throw UnsupportedError("quantType: $quantType");
    }
  }

  Tensor _readQ8(DataReader br, int rows, int cols, {bool fp32 = false}) {
    final q8 = Q8Tensor.readFrom(br, rows, cols);

    if (fp32) {
      return q8.toFP32Tensor(cached: false);
    }

    return q8;
  }

  Tensor _readQ16(DataReader br, int rows, int cols, {bool fp32 = false}) {
    final q16 = Q16Tensor.readFrom(br, rows, cols);

    if (fp32) {
      return q16.toFP32Tensor(cached: false);
    }

    return q16;
  }

  @override
  String toString() =>
      'ModelWeights{embedTokens: $embedTokens, layers: ${layers.length}, finalNorm: $finalNorm}';
}
