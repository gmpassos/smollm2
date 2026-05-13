import 'dart:math' as math;

import 'config.dart';
import 'data.dart';
import 'quant_type.dart';
import 'runtime.dart';
import 'tensor/tensor.dart';

class LayerWeights {
  final int index;

  LayerWeights(this.index);

  late FP32Tensor inputLayerNorm;

  late Tensor qProj;
  late Tensor kProj;
  late Tensor vProj;
  late Tensor oProj;

  late FP32Tensor postAttentionLayerNorm;

  late Tensor gateProj;
  late Tensor upProj;
  late Tensor downProj;

  List<Tensor> get tensors => [
    inputLayerNorm,
    qProj,
    kProj,
    vProj,
    oProj,
    postAttentionLayerNorm,
    gateProj,
    upProj,
    downProj,
  ];

  Map<Type, int> get tensorsTypes {
    final counts = <Type, int>{};

    for (final tensor in tensors) {
      counts.update(
        tensor.runtimeType,
        (count) => count + 1,
        ifAbsent: () => 1,
      );
    }

    return counts;
  }

  void load(
    Config config,
    DataReader dataReader,
    math.Random? jitterRandom,
    double? jitterScale,
  ) {
    final quantType = config.quantType;
    final hd = config.headDim;

    ////

    inputLayerNorm = FP32Tensor.readFrom(dataReader, config.hiddenSize);

    ////

    qProj = _readTensor(
      dataReader,
      config.numHeads * hd,
      config.hiddenSize,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
      quantType: quantType,
      asFP32: true,
    );

    kProj = _readTensor(
      dataReader,
      config.numKvHeads * hd,
      config.hiddenSize,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
      quantType: quantType,
      asFP32: true,
    );

    vProj = _readTensor(
      dataReader,
      config.numKvHeads * hd,
      config.hiddenSize,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
      quantType: quantType,
      asFP32: true,
    );

    oProj = _readTensor(
      dataReader,
      config.hiddenSize,
      config.numHeads * hd,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
      quantType: quantType,
      asFP32: true,
    );

    ////

    postAttentionLayerNorm = FP32Tensor.readFrom(dataReader, config.hiddenSize);

    ////

    gateProj = _readTensor(
      dataReader,
      config.intermediateSize,
      config.hiddenSize,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
      quantType: quantType,
      asFP32: true,
    );

    upProj = _readTensor(
      dataReader,
      config.intermediateSize,
      config.hiddenSize,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
      quantType: quantType,
      asFP32: true,
    );

    downProj = _readTensor(
      dataReader,
      config.hiddenSize,
      config.intermediateSize,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
      quantType: quantType,
      asFP32: true,
    );
  }
}

class ModelWeights {
  LoggerFunction? logger;

  final Config config;

  ModelWeights(this.config, {this.logger});

  void log(Object? o) {
    final logger = this.logger;
    if (logger != null) {
      logger(o);
    }
  }

  late Tensor embedTokens;
  late Float32ListX4 embedTokensArray;

  late List<LayerWeights> layers;
  late FP32Tensor finalNorm;

  Map<Type, int> get layersTensorsTypes => layers.first.tensorsTypes;

  void load(
    Config config,
    DataReader dataReader, {
    math.Random? jitterRandom,
    double? jitterScale,
  }) {
    if (jitterRandom != null) {
      jitterScale ??= config.quantType == QuantType.q8
          ? Q8Tensor.defaultQ8DequantizationJitterScale
          : Q16Tensor.defaultQ16DequantizationJitterScale;

      log(
        'Loading weights and dequantizing Q16 → FP32 with jitter injection '
        '(jitterScale: $jitterScale)...',
      );
    } else {
      log('Loading weights...');
    }

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
      asFP32: true,
    );

    embedTokensArray = embedTokens
        .toFP32Tensor(cached: true)
        .dataArray
        .asFloat32ListX4;

    log("├─ Loaded embed tokens ✅");
  }

  void _loadLayers(
    Config config,
    DataReader dataReader,
    math.Random? jitterRandom,
    double? jitterScale,
  ) {
    final numLayers = config.numLayers;

    log("├─ Loading $numLayers layers...");

    layers = List.generate(numLayers, (i) {
      final lw = LayerWeights(i);
      lw.load(config, dataReader, jitterRandom, jitterScale);

      i++;
      log(
        [
          "│  ",
          i == numLayers ? "└─" : "├─",
          " Loaded layer $i/$numLayers ✅ ${lw.tensorsTypes}",
        ].join(),
      );
      return lw;
    });

    log("├─ Loaded ${layers.length} layers ✅");
  }

  void _loadFinalNorm(Config config, DataReader dataReader) {
    finalNorm = FP32Tensor.readFrom(dataReader, config.hiddenSize);

    log("└─ Loaded final normalization layer ✅");
  }

  @override
  String toString() =>
      'ModelWeights{embedTokens: $embedTokens, layers: ${layers.length}, finalNorm: $finalNorm}';
}

final TensorFactory _tensorFactory = TensorFactory.instance..load();

Tensor _readTensor(
  DataReader br,
  int rows,
  int cols, {
  required QuantType quantType,
  bool asFP32 = false,
  math.Random? jitterRandom,
  double? jitterScale,
}) {
  return _tensorFactory.readTensor(quantType, br, rows, cols, asFP32: asFP32);
}
