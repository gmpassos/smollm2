import 'quant_type.dart';

class Config {
  final QuantType quantType;
  final int groupSize;
  final int hiddenSize;
  final int intermediateSize;
  final int numLayers;
  final int numHeads;
  final int numKvHeads;
  final int vocabSize;
  final int maxSeqLen;

  final double ropeTheta;
  final double rmsNormEps;

  final int headDim;

  Config({
    required this.quantType,
    required this.groupSize,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numLayers,
    required this.numHeads,
    required this.numKvHeads,
    required this.vocabSize,
    required this.maxSeqLen,
    required this.ropeTheta,
    required this.rmsNormEps,
    required this.headDim,
  });

  @override
  String toString() =>
      'Config{'
      'quantType: $quantType, '
      'groupSize: $groupSize, '
      'hiddenSize: $hiddenSize, '
      'intermediateSize: $intermediateSize, '
      'numLayers: $numLayers, '
      'numHeads: $numHeads, '
      'numKvHeads: $numKvHeads, '
      'vocabSize: $vocabSize, '
      'maxSeqLen: $maxSeqLen, '
      'headDim: $headDim'
      '}';
}
