import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'activations.dart';
import 'config.dart';
import 'data.dart';
import 'kv_cache.dart';
import 'quant_type.dart';
import 'tensor/native/macos/cpu/fp32_cpu_tensor.dart';
import 'tensor/native/macos/metal/fp32_metal_tensor.dart';
import 'tensor/native/windows/cuda/fp32_cuda_tensor.dart';
import 'tensor/tensor.dart';
import 'tokenizer.dart';
import 'weights.dart';

typedef LoggerFunction = void Function(Object?)?;

class LLMRuntime {
  LoggerFunction? logger;

  LLMRuntime({this.logger});

  void log(Object? o) {
    final logger = this.logger;
    if (logger != null) {
      logger(o);
    }
  }

  late final Config config;

  late final Tokenizer tokenizer;
  late final TokenizerEngine tokenizerEngine;

  late final ModelWeights weights;

  late final List<KVCache> kvCaches;

  late final TensorFloat32Data x;
  late final TensorFloat32Data xb;
  late final TensorFloat32Data xb2;

  late final TensorFloat32Data q;
  late final Float32List qViewBuffer;
  late final TensorFloat32Data k;
  late final Float32List kViewBuffer;
  late final TensorFloat32Data v;
  late final Float32List vViewBuffer;

  late final Float32List attention;
  late final double attentionScale;

  late final Float32List logits;

  late final TensorFloat32Data hb;
  late final Float32List hbViewBuffer;
  late final TensorFloat32Data hb2;
  late final Float32List hb2ViewBuffer;

  bool _loaded = false;

  /// Whether the model has been successfully loaded and is ready for runtime inference.
  bool get isLoaded => _loaded;

  /// Loads a model from [modelPath] and prepares it for inference.
  ///
  /// Supported model formats:
  /// - BF16: loaded directly as BF16 weights and converted to FP32 internally.
  /// - Q8: loaded as 8-bit quantized weights and dequantized to FP32.
  /// - Q16: loaded as 16-bit quantized weights and dequantized to FP32.
  ///
  /// During loading, optional controlled dequantization jitter can be applied
  /// to quantized models (Q8/Q16) to introduce small stochastic variations in
  /// the reconstructed weights. This may reduce quantization artifacts but can
  /// introduce slight nondeterminism depending on configuration.
  ///
  /// Jitter is considered enabled internally when [jitterRandom],
  /// [jitterScale], or [jitterSeed] is provided, unless explicitly disabled
  /// via [applyDequantizationJitter].
  ///
  /// - [modelPath]: Path to the model file.
  ///
  /// - [applyDequantizationJitter]: Explicitly enables or disables jitter
  ///   during dequantization. If null, it is automatically enabled when either
  ///   [jitterRandom], [jitterScale], or [jitterSeed] is provided.
  ///
  /// - [jitterSeed]: Optional seed used to initialize the random generator
  ///   when jitter is enabled and [jitterRandom] is not provided.
  ///
  /// - [jitterRandom]: Optional random generator used to produce deterministic
  ///   jitter when seeded. If null while jitter is enabled, a default RNG is
  ///   created internally.
  ///
  /// - [jitterScale]: Controls the magnitude of the injected jitter. Typical
  ///   values are small (e.g. 0.01). If null, a default scale is used.
  Future<void> load(
    String modelPath, {
    bool? applyDequantizationJitter,
    int? jitterSeed,
    math.Random? jitterRandom,
    double? jitterScale,
  }) async {
    if (_loaded) return;
    _loaded = true;

    var modelFile = File(modelPath);

    var modelFileLength = await modelFile.length();

    log('Loading model: $modelPath ($modelFileLength bytes) ...');

    applyDequantizationJitter ??=
        jitterRandom != null || jitterScale != null || jitterSeed != null;

    if (applyDequantizationJitter) {
      // Ensure that jitter is enabled:
      jitterRandom ??= math.Random(jitterSeed);
    } else {
      // Disable jitter:
      jitterRandom = null;
    }

    final loadStart = Stopwatch()..start();

    final modelIO = await modelFile.open();
    final dataReader = DataReader(modelIO);

    final quantType = _loadHeader(dataReader);

    _loadConfig(dataReader, quantType);
    _loadTokenizer(dataReader);
    _loadWeights(dataReader, jitterRandom, jitterScale);

    await modelIO.close();

    _precomputeRoPE();
    _buildKVCaches();

    final hd = config.headDim;
    final numHeads = config.numHeads;
    final hiddenSize = config.hiddenSize;
    final numKvHeads = config.numKvHeads;
    final intermediateSize = config.intermediateSize;
    final attSize = config.maxSeqLen;
    final vocabSize = config.vocabSize;

    var layersTensorsTypes = weights.layersTensorsTypes;
    // print('!!! $layersTensorsTypes: $layersTensorsTypes');

    TensorFloat32DataCreator tensorCreator;
    if (layersTensorsTypes.containsKey(FP32CPUTensor)) {
      tensorCreator = CPUTensorFloat32Data.new;
    } else if (layersTensorsTypes.containsKey(FP32MetalTensor)) {
      tensorCreator = MetalTensorFloat32Data.new;
    } else if (layersTensorsTypes.containsKey(FP32CudaTensor)) {
      tensorCreator = CudaTensorFloat32Data.new;
    } else {
      tensorCreator = TensorFloat32Data.new;
    }

    x = TensorFloat32Data(hiddenSize);
    xb = tensorCreator(hiddenSize);
    xb2 = TensorFloat32Data(hiddenSize);

    q = tensorCreator(numHeads * hd);
    qViewBuffer = Float32List(q.length);

    k = tensorCreator(numKvHeads * hd);
    kViewBuffer = Float32List(k.length);

    v = tensorCreator(numKvHeads * hd);
    vViewBuffer = Float32List(v.length);

    attention = Float32List(attSize * numHeads);
    attentionScale = 1.0 / math.sqrt(hd.toDouble());

    logits = Float32List(vocabSize);

    hb = tensorCreator(intermediateSize);
    hbViewBuffer = Float32List(intermediateSize);

    hb2 = tensorCreator(intermediateSize);
    hb2ViewBuffer = Float32List(intermediateSize);

    loadStart.stop();

    log(
      'Model fully loaded 🚀 (${loadStart.elapsed.formattedToHumanReadable})',
    );
  }

  ///// LOAD FUNCTIONS /////

  QuantType _loadHeader(DataReader dataReader) {
    final magic = utf8.decode(dataReader.readBytes(4));
    if (magic != 'SMOL') {
      throw Exception('Bad magic');
    }

    final ver = dataReader.readU32();

    if (ver == 1) {
      return QuantType.q8;
    } else if (ver == 2) {
      final qt = dataReader.readU32();
      dataReader.readU32();
      final quantType = QuantType.withValue(qt);
      if (quantType != QuantType.q8 &&
          quantType != QuantType.q16 &&
          quantType != QuantType.bf16) {
        throw Exception('Q8, Q16 and BF16 only');
      }
      return quantType;
    }

    throw StateError('Bad version: $ver');
  }

  void _loadConfig(DataReader dataReader, QuantType quantType) {
    final loadStart = Stopwatch()..start();

    final hiddenSize = dataReader.readU32();
    final intermediateSize = dataReader.readU32();
    final numLayers = dataReader.readU32();
    final numHeads = dataReader.readU32();
    final numKvHeads = dataReader.readU32();
    final vocabSize = dataReader.readU32();
    final maxSeqLen = dataReader.readU32();
    final ropeTheta = dataReader.readF32();
    final rmsNormEps = dataReader.readF32();

    final headDim = hiddenSize ~/ numHeads;

    config = Config(
      quantType: quantType,
      groupSize: 0,
      hiddenSize: hiddenSize,
      intermediateSize: intermediateSize,
      numLayers: numLayers,
      numHeads: numHeads,
      numKvHeads: numKvHeads,
      vocabSize: vocabSize,
      maxSeqLen: maxSeqLen,
      ropeTheta: ropeTheta,
      rmsNormEps: rmsNormEps,
      headDim: headDim,
    );

    loadStart.stop();

    log('Loaded (${loadStart.elapsed.formattedToHumanReadable}): $config');
  }

  void _loadTokenizer(DataReader dataReader) {
    final loadStart = Stopwatch()..start();

    tokenizer = Tokenizer.loadFrom(dataReader);
    tokenizerEngine = TokenizerEngine(tokenizer);

    loadStart.stop();

    log('Loaded (${loadStart.elapsed.formattedToHumanReadable}): $tokenizer');
  }

  void _loadWeights(
    DataReader dataReader,
    math.Random? jitterRandom,
    double? jitterScale,
  ) {
    final loadStart = Stopwatch()..start();

    weights = ModelWeights(config, logger: logger);

    weights.load(
      config,
      dataReader,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
    );

    loadStart.stop();

    log('Loaded (${loadStart.elapsed.formattedToHumanReadable}): $weights');
  }

  ///// BUILD /////

  late Float32List ropeCos; // [maxSeqLen * headDim/2]
  late Float32List ropeSin;

  void _precomputeRoPE() {
    final hd = config.headDim;
    final maxLen = config.maxSeqLen;
    final theta = config.ropeTheta;

    ropeCos = Float32List(maxLen * hd);
    ropeSin = Float32List(maxLen * hd);

    for (int p = 0; p < maxLen; p++) {
      for (int i = 0; i < hd ~/ 2; i++) {
        final freq = 1.0 / math.pow(theta, (2 * i) / hd);
        final c = math.cos(p * freq);
        final s = math.sin(p * freq);

        ropeCos[p * hd + i] = c;
        ropeCos[p * hd + i + hd ~/ 2] = c;

        ropeSin[p * hd + i] = s;
        ropeSin[p * hd + i + hd ~/ 2] = s;
      }
    }
  }

  void _buildKVCaches() {
    kvCaches = List.generate(
      config.numLayers,
      (_) => KVCache.fromConfig(config),
    );
  }

  /// Resets all KV caches and clears token generation state.
  ///
  /// This invalidates all cached attention keys and values accumulated from
  /// previous inference steps, forcing subsequent generations to rebuild the
  /// context from scratch.
  ///
  /// Also resets the internal processed token counter.
  void resetCache() {
    for (final kv in kvCaches) {
      kv.cacheLen = 0;
    }

    _totalTokens = 0;
  }

  /// Total number of tokens processed by this [LLMRuntime] instance since the last
  /// [resetCache] call.
  ///
  /// This counter is cumulative and monotonic: every ingested prompt token and
  /// every generated token increments this value (by [forward]).
  ///
  /// Unlike [contextTokens], this value is not limited by the active context
  /// window and continues growing for the full lifetime of the inference session.
  int get totalTokens => _totalTokens;

  int _totalTokens = 0;

  ///////////////////////////////////////////

  Float32ListX4 _outoutBuffer = Float32List(0).asFloat32ListX4;

  void rmsNorm(
    final TensorFloat32Data out,
    final TensorFloat32Data x,
    final TensorFloat32Data w,
    final int n,
    final double eps,
  ) {
    if (n == 0) return;

    // print('!!! ${out.runtimeType} ; ${x.runtimeType} ; ${w.runtimeType}');

    final xArray = x.array;
    final xArrayX4 = x.arrayX4;

    final wArray = w.array;
    final wArrayX4 = w.arrayX4;

    final n4 = n ~/ 4;

    Float32x4 ssVec;
    {
      final v = xArrayX4[0];
      ssVec = v * v;
    }

    // -------------------------
    // sum of squares (SIMD)
    // -------------------------
    for (int i = 1; i < n4; i++) {
      final v = xArrayX4[i];
      ssVec += v * v;
    }

    double ss = ssVec.x + ssVec.y + ssVec.z + ssVec.w;

    // tail (if n not multiple of 4)
    for (int i = n4 << 2; i < n; i++) {
      final v = xArray[i];
      ss += v * v;
    }

    final scale = 1.0 / math.sqrt((ss / n) + eps);

    final scaleVec = Float32x4.splat(scale);

    final Float32List outArray;
    final Float32x4List outArrayX4;
    final bool needsCopyToOut;

    if (out.dataType == Float32List) {
      outArray = out.array;
      outArrayX4 = out.arrayX4;
      needsCopyToOut = false;
    } else {
      if (_outoutBuffer.length != out.length) {
        _outoutBuffer = Float32List(out.length).asFloat32ListX4;
      }

      outArray = _outoutBuffer.list;
      outArrayX4 = _outoutBuffer.listX4;
      needsCopyToOut = true;
    }

    // -------------------------
    // normalize (SIMD)
    // -------------------------
    for (int i = 0; i < n4; i++) {
      outArrayX4[i] = xArrayX4[i] * scaleVec * wArrayX4[i];
    }

    // -------------------------
    // tail (scalar)
    // -------------------------
    for (int i = n4 << 2; i < n; i++) {
      outArray[i] = xArray[i] * scale * wArray[i];
    }

    if (needsCopyToOut) {
      out.setRange(0, out.length, outArray, 0);
    }
  }

  void applyRoPE(Float32List v, int offset, int hd, int pos) {
    final half = hd ~/ 2;
    final base = pos * hd;

    for (int i = 0; i < half; i++) {
      final offsetI = offset + i;
      final offsetIHalf = offsetI + half;

      final v0 = v[offsetI];
      final v1 = v[offsetIHalf];

      final c = ropeCos[base + i];
      final s = ropeSin[base + i];

      v[offsetI] = v0 * c - v1 * s;
      v[offsetIHalf] = v1 * c + v0 * s;
    }
  }

  void forward(final int tok) {
    final pos = _totalTokens++;
    final c = config;
    final w = weights;

    final hs = c.hiddenSize;
    final nh = c.numHeads;
    final nkv = c.numKvHeads;
    final ng = nh ~/ nkv;

    final xArray = x.array;
    final xArrayX4 = x.arrayX4;

    // Embedding:
    {
      final embOffset = tok * hs;
      xArray.setRange(0, hs, w.embedTokensArray.list, embOffset);
    }

    for (int l = 0; l < c.numLayers; l++) {
      _forwardLayer(l, pos, nkv, ng);
    }

    // -------------------------
    // Final norm
    // -------------------------
    rmsNorm(x, x, w.finalNorm.data, hs, c.rmsNormEps);

    // -------------------------
    // Logits
    // -------------------------
    _computeLogits(xArray, xArrayX4);
  }

  void _computeLogits(Float32List xArray, Float32x4List xArrayX4) {
    final c = config;
    final w = weights;
    final hs = c.hiddenSize;

    final hsX4 = hs ~/ 4;
    final tailStart = hsX4 * 4;

    final logits = this.logits;

    final emb = w.embedTokensArray.list;
    final embX4 = w.embedTokensArray.listX4;
    final zeroX4 = Float32x4.zero();

    for (int i = 0; i < c.vocabSize; i++) {
      final row = i * hs;
      final rowX4 = i * hsX4;

      var sumX4 = zeroX4;

      for (int j = 0; j < hsX4; j++) {
        sumX4 += xArrayX4[j] * embX4[rowX4 + j];
      }

      double sum = sumX4.x + sumX4.y + sumX4.z + sumX4.w;

      for (int j = tailStart; j < hs; j++) {
        sum += xArray[j] * emb[row + j];
      }

      logits[i] = sum;
    }
  }

  void _forwardLayer(int l, int pos, int nkv, int ng) {
    final c = config;
    final w = weights;

    final nh = c.numHeads;
    final nkv = c.numKvHeads;
    final ng = nh ~/ nkv;

    ////////////////////

    final layer = w.layers[l];
    final kv = kvCaches[l];

    final cacheLen = kv.cacheLen;
    final slen = cacheLen + 1;
    final attSize = c.maxSeqLen;

    ////////////////////

    _rmsNormInput(layer);

    _qkvProjection(layer);

    _applyRoPEToQK(pos);

    _kvCacheWrite(kv, cacheLen);

    _computeSelfAttention(kv, ng, attSize, slen, pos);

    _outputProjection(layer);

    _computeMLP(layer);

    kv.cacheLen = slen;
  }

  void _rmsNormInput(LayerWeights layer) {
    final Config c = config;
    rmsNorm(xb, x, layer.inputLayerNorm.data, c.hiddenSize, c.rmsNormEps);
  }

  void _qkvProjection(LayerWeights layer) {
    final xb = this.xb;
    layer.qProj.dotTo(q, xb);
    layer.kProj.dotTo(k, xb);
    layer.vProj.dotTo(v, xb);
  }

  void _applyRoPEToQK(int pos) {
    final Config c = config;
    final hd = c.headDim;
    final nh = c.numHeads;
    final nkv = c.numKvHeads;

    final qArray = q.arrayView(buffer: qViewBuffer);
    final kArray = k.arrayView(buffer: kViewBuffer);

    for (int h = 0; h < nh; h++) {
      applyRoPE(qArray, h * hd, hd, pos);
    }

    for (int h = 0; h < nkv; h++) {
      applyRoPE(kArray, h * hd, hd, pos);
    }

    if (!q.isInDartMemory) {
      q.setRange(0, q.length, qArray);
    }

    if (!k.isInDartMemory) {
      k.setRange(0, k.length, kArray);
    }
  }

  void _kvCacheWrite(KVCache kv, int cacheLen) {
    final c = config;
    final hd = c.headDim;
    final nkv = c.numKvHeads;

    ////////////////////

    final kCache = kv.kCache;
    final vCache = kv.vCache;
    final cacheLen = kv.cacheLen;

    final kArray = k.arrayView(buffer: kViewBuffer);
    final vArray = v.arrayView(buffer: vViewBuffer);

    for (int h = 0; h < nkv; h++) {
      final dst = kv.offset(h, cacheLen);
      final src = h * hd;

      for (int i = 0; i < hd; i++) {
        kCache[dst + i] = kArray[src + i];
        vCache[dst + i] = vArray[src + i];
      }
    }
  }

  void _computeSelfAttention(
    KVCache kv,
    int ng,
    int attSize,
    int slen,
    int pos,
  ) {
    final c = config;

    final hs = c.hiddenSize;
    final hd = c.headDim;
    final nh = c.numHeads;
    final nkv = c.numKvHeads;
    final ng = nh ~/ nkv;

    final xb2Array = xb2.array;

    final kCache = kv.kCache;
    final vCache = kv.vCache;

    final qArray = q.arrayView(buffer: qViewBuffer);

    ////////////////////

    final attSize = c.maxSeqLen;

    xb2Array.fillRange(0, hs, 0.0);

    final attention = this.attention;
    final attentionScale = this.attentionScale;

    for (int h = 0; h < nh; h++) {
      final kvh = h ~/ ng;
      final qOff = h * hd;

      final cacheBase = kvh * attSize * hd;
      final attBase = h * attSize;

      // compute scores
      for (int t = 0; t < slen; t++) {
        if (t > pos) {
          attention[attBase + t] = -1e9;
        } else {
          final kOff = cacheBase + t * hd;
          double score = 0.0;

          for (int d = 0; d < hd; d++) {
            score += qArray[qOff + d] * kCache[kOff + d];
          }

          attention[attBase + t] = score * attentionScale;
        }
      }

      softmaxSegment(attention, attBase, slen);

      final outOffset = h * hd;

      for (int t = 0; t < slen; t++) {
        final a = attention[attBase + t];
        final vOff = cacheBase + t * hd;

        for (int d = 0; d < hd; d++) {
          xb2Array[outOffset + d] += a * vCache[vOff + d];
        }
      }
    }
  }

  void _outputProjection(LayerWeights layer) {
    final c = config;
    final hs = c.hiddenSize;

    layer.oProj.dotTo(xb, xb2);

    xb.addTo(x.array, hs);
  }

  void _computeMLP(LayerWeights layer) {
    final c = config;
    final hs = c.hiddenSize;

    final x = this.x;
    final xArray = x.array;

    final xb = this.xb;
    final hb = this.hb;
    final hb2 = this.hb2;

    rmsNorm(xb, x, layer.postAttentionLayerNorm.data, hs, c.rmsNormEps);

    layer.gateProj.dotTo(hb, xb);
    layer.upProj.dotTo(hb2, xb);

    final hbArray = hb.arrayView(buffer: hbViewBuffer);
    final hb2Array = hb2.arrayView(buffer: hb2ViewBuffer);
    final inter = c.intermediateSize;

    for (int i = 0; i < inter; i++) {
      hbArray[i] = silu(hbArray[i]) * hb2Array[i];
    }

    if (!hb.isInDartMemory) {
      hb.setRange(0, hb.length, hbArray);
    }

    layer.downProj.dotTo(xb, hb);

    xb.addTo(xArray, hs);
  }

  int sample(
    int vocab,
    double temperature,
    double repeatPenalty,
    Map<int, int> seen,
    math.Random rand, {
    bool eager = false,
  }) {
    final logits = this.logits;

    // Apply repetition penalty (if enabled):
    // - skips work when repeatPenalty == 1.0 (no effect)
    // - penalizes tokens based on how many times they were seen
    // - stronger repetition => stronger penalty (exponential scaling)
    // - reduces likelihood of repeated tokens in sampling
    if (repeatPenalty != 1.0) {
      for (int i = 0; i < vocab; i++) {
        final count = seen[i] ?? 0;

        if (count > 0) {
          final factor = math.pow(repeatPenalty, count).toDouble();
          final logit = logits[i];

          logits[i] = logit > 0
              ?
                // If token is currently likely, reduce its score
                // so repeated tokens become less probable
                logit / factor
              :
                // If token is already unlikely, make it even less likely
                // by increasing its magnitude in the negative direction
                logit * factor;
        }
      }
    }

    // Greedy / deterministic mode:
    if (eager || temperature <= 0.0) {
      int best = 0;

      for (int i = 1; i < vocab; i++) {
        if (logits[i] > logits[best]) {
          best = i;
        }
      }

      return best;
    }
    // Stochastic sampling mode:
    // - converts logits into a probability distribution (softmax)
    // - samples a token using cumulative probability (roulette wheel)
    else {
      // Applies temperature scaling:
      for (int i = 0; i < vocab; i++) {
        logits[i] /= temperature;
      }

      softmax(logits, vocab);

      final r = rand.nextDouble();
      double acc = 0.0;

      for (int i = 0; i < vocab; i++) {
        acc += logits[i];
        if (acc >= r) {
          return i;
        }
      }

      return vocab - 1;
    }
  }

  List<int> tokenize(String text, int max) {
    return tokenizerEngine.tokenize(text, max);
  }

  String decode(int tok) {
    return tokenizerEngine.decode(tok);
  }
}
