import 'dart:typed_data';

import 'config.dart';

class KVCache {
  final int numKvHeads;
  final int maxSeqLen;
  final int headDim;

  final Float32List kCache;
  final Float32List vCache;

  int cacheLen = 0;

  KVCache({
    required this.numKvHeads,
    required this.maxSeqLen,
    required this.headDim,
  }) : kCache = Float32List(numKvHeads * maxSeqLen * headDim),
       vCache = Float32List(numKvHeads * maxSeqLen * headDim);

  factory KVCache.fromConfig(Config config) => KVCache(
    numKvHeads: config.numKvHeads,
    maxSeqLen: config.maxSeqLen,
    headDim: config.headDim,
  );
}
