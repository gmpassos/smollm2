import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'activations.dart';
import 'config.dart';
import 'data.dart';
import 'kv_cache.dart';
import 'quant_type.dart';
import 'tensor.dart';
import 'token_generator.dart';
import 'tokenizer.dart';
import 'weights.dart';

/// SmolLM2 model loader and runtime interface.
///
/// Responsible for loading a quantized model (Q16) and preparing it
/// for inference in FP32, optionally applying controlled jitter during
/// dequantization to improve numerical robustness or variability.
class SmolLM2 implements TokenGenerator {
  final void Function(Object?)? logger;

  SmolLM2({this.logger});

  void log(Object? o) {
    final logger = this.logger;
    if (logger != null) {
      logger(o);
    }
  }

  late final Config config;
  late final Tokenizer tokenizer;

  late final ModelWeights weights;

  late final Float32List ropeCos;
  late final Float32List ropeSin;

  late final List<KVCache> kvCaches;

  late final Float32ListX4 x;
  late final Float32ListX4 xb;
  late final Float32ListX4 xb2;

  late final Float32List q;
  late final Float32List k;
  late final Float32List v;

  late final Float32List att;
  late final Float32List logits;

  late final Float32List hb;
  late final Float32List hb2;

  bool _loaded = false;

  /// Whether the model has been successfully loaded and is ready for inference.
  bool get isLoaded => _loaded;

  /// Loads a SmolLM2 model from [modelPath] and prepares it for inference.
  ///
  /// This process reads the quantized weights (Q16) and converts them to FP32.
  /// Optionally, a controlled dequantization jitter can be applied to introduce
  /// small stochastic variations in the weights. This may reduce quantization
  /// artifacts but can introduce slight nondeterminism depending on configuration.
  ///
  /// Jitter is considered enabled internally when [jitterRandom] and/or
  /// [jitterScale] is provided, unless explicitly disabled via
  /// [applyDequantizationJitter].
  ///
  /// - [modelPath]: Path to the model file.
  ///
  /// - [applyDequantizationJitter]: Explicitly enables or disables jitter
  ///   during dequantization. If null, it is automatically enabled when either
  ///   [jitterRandom] or [jitterScale] is provided.
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

    _precomputeRope();
    _buildKVCaches();

    final hd = config.headDim;

    x = Float32List(config.hiddenSize).asFloat32ListX4;
    xb = Float32List(config.hiddenSize).asFloat32ListX4;
    xb2 = Float32List(config.hiddenSize).asFloat32ListX4;

    q = Float32List(config.numHeads * hd);
    k = Float32List(config.numKvHeads * hd);
    v = Float32List(config.numKvHeads * hd);

    att = Float32List(config.maxSeqLen * config.numHeads);
    logits = Float32List(config.vocabSize);

    hb = Float32List(config.intermediateSize);
    hb2 = Float32List(config.intermediateSize);

    loadStart.stop();

    log(
      'Model fully loaded 🚀 (${loadStart.elapsed.formattedToHumanReadable})',
    );
  }

  void _buildKVCaches() {
    kvCaches = List.generate(
      config.numLayers,
      (_) => KVCache.fromConfig(config),
    );
  }

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

    loadStart.stop();

    log('Loaded (${loadStart.elapsed.formattedToHumanReadable}): $tokenizer');
  }

  void _loadWeights(
    DataReader dataReader,
    math.Random? jitterRandom,
    double? jitterScale,
  ) {
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

    final loadStart = Stopwatch()..start();

    weights = ModelWeights(config);

    weights.load(
      config,
      dataReader,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
    );

    loadStart.stop();

    log('Loaded (${loadStart.elapsed.formattedToHumanReadable}): $weights');
  }

  late Float32List ropeCos; // [maxSeqLen * headDim/2]
  late Float32List ropeSin;

  void _precomputeRope() {
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

  ///////////////////////////////////////////

  void rmsNorm(
    final Float32ListX4 out,
    final Float32ListX4 x,
    final Float32ListX4 w,
    final int n,
    final double eps,
  ) {
    if (n == 0) return;

    final outList = out.list;
    final outListX4 = out.listX4;

    final xList = x.list;
    final xListX4 = x.listX4;

    final wList = w.list;
    final wList4 = w.listX4;

    final n4 = n ~/ 4;

    Float32x4 ssVec;
    {
      final v = xListX4[0];
      ssVec = v * v;
    }

    // -------------------------
    // sum of squares (SIMD)
    // -------------------------
    for (int i = 1; i < n4; i++) {
      final v = xListX4[i];
      ssVec += v * v;
    }

    double ss = ssVec.x + ssVec.y + ssVec.z + ssVec.w;

    // tail (if n not multiple of 4)
    for (int i = n4 << 2; i < n; i++) {
      final v = xList[i];
      ss += v * v;
    }

    final scale = 1.0 / math.sqrt((ss / n) + eps);

    final scaleVec = Float32x4.splat(scale);

    // -------------------------
    // normalize (SIMD)
    // -------------------------
    for (int i = 0; i < n4; i++) {
      outListX4[i] = xListX4[i] * scaleVec * wList4[i];
    }

    // -------------------------
    // tail (scalar)
    // -------------------------
    for (int i = n4 << 2; i < n; i++) {
      outList[i] = xList[i] * scale * wList[i];
    }
  }

  void applyRope(Float32List v, int offset, int hd, int pos) {
    final half = hd ~/ 2;
    final base = pos * hd;

    for (int i = 0; i < half; i++) {
      final v0 = v[offset + i];
      final v1 = v[offset + i + half];

      final c = ropeCos[base + i];
      final s = ropeSin[base + i];

      v[offset + i] = v0 * c - v1 * s;
      v[offset + i + half] = v1 * c + v0 * s;
    }
  }

  void forward(final int tok) {
    final pos = _totalTokens++;
    _contextTokens++;

    final c = config;
    final w = weights;

    final hs = c.hiddenSize;
    final hd = c.headDim;
    final nh = c.numHeads;
    final nkv = c.numKvHeads;
    final ng = nh ~/ nkv;

    final emb = w.embedTokens.toFP32Tensor(cached: true).data.list;

    final x = this.x;
    final xList = x.list;

    final xb = this.xb;
    final xbList = xb.list;

    final xb2 = this.xb2;
    final xb2List = xb2.list;

    final att = this.att;

    {
      final embOffset = tok * hs;

      // -------------------------
      // Embedding
      // -------------------------
      xList.setRange(0, hs, emb, embOffset);
    }

    final scale = 1.0 / math.sqrt(hd.toDouble());

    // -------------------------
    // Layers
    // -------------------------
    for (int l = 0; l < c.numLayers; l++) {
      final lw = w.layers[l];
      final kv = kvCaches[l];

      final kCache = kv.kCache;
      final vCache = kv.vCache;

      final slen = kv.cacheLen + 1;
      final attSize = c.maxSeqLen;

      // -------------------------
      // RMSNorm (input)
      // -------------------------
      rmsNorm(xb, x, lw.inputLayerNorm.data, hs, c.rmsNormEps);

      // -------------------------
      // QKV projections
      // -------------------------
      lw.qProj.dotTo(q, xb);
      lw.kProj.dotTo(k, xb);
      lw.vProj.dotTo(v, xb);

      // -------------------------
      // RoPE
      // -------------------------

      {
        final ropeOffset = pos * hd;
        for (int h = 0; h < nh; h++) {
          applyRope(q, h * hd, hd, ropeCos, ropeSin, ropeOffset);
        }
        for (int h = 0; h < nkv; h++) {
          applyRope(k, h * hd, hd, ropeCos, ropeSin, ropeOffset);
        }
      }

      // -------------------------
      // KV cache write (FAST)
      // -------------------------
      {
        final seqStride = c.maxSeqLen * hd;
        final dstOff = kv.cacheLen * hd;

        var src = 0;
        var dst = dstOff;

        for (int h = 0; h < nkv; h++) {
          var end = dst + hd;
          kCache.setRange(dst, end, k, src);
          vCache.setRange(dst, end, v, src);

          src += hd;
          dst += seqStride;
        }
      }

      // -------------------------
      // Attention
      // -------------------------
      xb2List.fillRange(0, hs, 0.0);

      for (int h = 0; h < nh; h++) {
        final kvh = h ~/ ng;

        final qOff = h * hd;
        final cacheBase = kvh * c.maxSeqLen * hd;
        final attBase = h * attSize;

        final q = this.q;
        final k = kCache;
        final v = vCache;

        for (int t = 0; t < slen; t++) {
          final kOff = cacheBase + t * hd;

          double sum = 0.0;

          // unrolled dot (fast in Dart VM)
          for (int d = 0; d < hd; d += 4) {
            final i = qOff + d;
            final j = kOff + d;

            sum +=
                q[i] * k[j] +
                q[i + 1] * k[j + 1] +
                q[i + 2] * k[j + 2] +
                q[i + 3] * k[j + 3];
          }

          att[attBase + t] = sum * scale;
        }

        softmaxSegment(att, attBase, slen);

        final outOff = h * hd;

        for (int t = 0; t < slen; t++) {
          final a = att[attBase + t];
          final vOff = cacheBase + t * hd;

          for (int d = 0; d < hd; d++) {
            xb2List[outOff + d] += a * v[vOff + d];
          }
        }
      }

      // -------------------------
      // Output projection
      // -------------------------
      lw.oProj.dotTo(xbList, xb2);

      for (int i = 0; i < hs; i++) {
        xList[i] += xbList[i];
      }

      // -------------------------
      // MLP
      // -------------------------
      rmsNorm(xb, x, lw.postAttentionLayerNorm.data, hs, c.rmsNormEps);

      lw.gateProj.dotTo(this.hb, xb);
      lw.upProj.dotTo(this.hb2, xb);

      final hb = this.hb;
      final hb2 = this.hb2;
      final inter = c.intermediateSize;

      for (int i = 0; i < inter; i++) {
        hb[i] = silu(hb[i]) * hb2[i];
      }

      lw.downProj.dotTo(xbList, hb.asFloat32ListX4);

      for (int i = 0; i < hs; i++) {
        xList[i] += xbList[i];
      }

      kv.cacheLen = slen;
    }

    // -------------------------
    // Final norm
    // -------------------------
    rmsNorm(x, x, w.finalNorm.data, hs, c.rmsNormEps);

    // -------------------------
    // Logits
    // -------------------------
    final embF = w.embedTokens.toFP32Tensor(cached: true).data.list;
    final logits = this.logits;

    for (int i = 0; i < c.vocabSize; i++) {
      double sum = 0.0;
      final row = i * hs;

      for (int j = 0; j < hs; j += 4) {
        sum +=
            xList[j] * embF[row + j] +
            xList[j + 1] * embF[row + j + 1] +
            xList[j + 2] * embF[row + j + 2] +
            xList[j + 3] * embF[row + j + 3];
      }

      logits[i] = sum;
    }
  }

  late final _tokenizerEngine = TokenizerEngine(tokenizer);

  List<int> tokenize(String text, int max) {
    return _tokenizerEngine.tokenize(text, max);
  }

  String decode(int tok) {
    return _tokenizerEngine.decode(tok);
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

  static final _randomSecure = math.Random.secure();

  static int generateSeed() {
    return _randomSecure.nextInt(0x7fffffff);
  }

  /// Total number of tokens processed by this model instance since the last
  /// [resetCache] call.
  ///
  /// This counter is cumulative and monotonic: every ingested prompt token and
  /// every generated token increments this value.
  ///
  /// Unlike [contextTokens], this value is not limited by the active context
  /// window and continues growing for the full lifetime of the inference session.
  int get totalTokens => _totalTokens;

  int _totalTokens = 0;

  /// Number of token positions currently resident in the active transformer
  /// context (KV cache).
  ///
  /// This represents how many past tokens are presently available for attention
  /// during the next forward pass.
  ///
  /// Unlike [totalTokens], this value is bounded by the model's maximum sequence
  /// length and may be lower if the cache is reset, truncated, or compacted.
  int get contextTokens => _contextTokens;

  int _contextTokens = 0;

  void resetCache() {
    for (final kv in kvCaches) {
      kv.cacheLen = 0;
    }

    _totalTokens = 0;
    _contextTokens = 0;
  }

  StringBuffer? _fullText;

  String get fullText => _fullText?.toString() ?? '';

  int get fullTextLength => _fullText?.length ?? 0;

  Map<int, int>? _seen;

  void reset() {
    resetCache();
    _seen = null;
    _fullText = null;
  }

  Future<Duration> ingest(
    String prompt, {
    bool emmitPromptTokens = true,
    OnTokenEmitted? onTokenEmitted,
  }) async {
    var r = await _ingestImpl(
      prompt,
      emmitPromptTokens: emmitPromptTokens,
      onTokenEmitted: onTokenEmitted,
    );
    return r.$2;
  }

  Future<(List<int> toks, Duration promptDuration)> _ingestImpl(
    String prompt, {
    bool emmitPromptTokens = true,
    OnTokenEmitted? onTokenEmitted,
  }) async {
    final toks = tokenize(prompt, 512);
    if (toks.isEmpty) {
      throw StateError('Tokenize failed');
    }

    final StringBuffer fullText;
    final Map<int, int> seen;
    if (_seen == null) {
      resetCache();
      fullText = _fullText = StringBuffer();
      seen = _seen = <int, int>{};
    } else {
      fullText = _fullText!;
      seen = _seen!;
    }

    // Prompt ingestion timing (optional separate metric)
    final promptStart = Stopwatch()..start();

    for (int i = 0; i < toks.length; i++) {
      final tok = toks[i];
      forward(tok);
      seen.increment(tok);

      var s = decode(tok);
      fullText.write(s);

      if (onTokenEmitted != null && emmitPromptTokens) {
        onTokenEmitted(tok, s, TokenOrigin.prompt);
      }
    }

    promptStart.stop();
    final promptDuration = promptStart.elapsed;

    return (toks, promptDuration);
  }

  @override
  Future<TokenGenerationResult> generate(
    String prompt, {
    int maxTokens = TokenGenerator.defaultMaxTokens,
    double temperature = TokenGenerator.defaultTemperature,
    double repeatPenalty = TokenGenerator.defaultRepeatPenalty,
    int? seed,
    bool emmitPromptTokens = true,
    bool includePromptInOutput = true,
    OnTokenEmitted? onTokenEmitted,
    math.Random? random,
  }) async {
    seed ??= generateSeed();
    random ??= math.Random(seed);

    var output = StringBuffer();
    if (includePromptInOutput) {
      output.write(prompt);
    }

    final (toks, promptDuration) = await _ingestImpl(
      prompt,
      onTokenEmitted: onTokenEmitted,
      emmitPromptTokens: emmitPromptTokens,
    );

    final fullText = _fullText!;
    final seen = _seen!;

    var stopReason = TokenGenerationStopReason.maxTokensReached;
    int generatedTokens = 0;

    final genWatch = Stopwatch()..start();

    for (int i = 0; i < maxTokens; i++) {
      final next = sample(
        config.vocabSize,
        temperature,
        repeatPenalty,
        seen,
        random,
      );

      if (next == 2) {
        stopReason = TokenGenerationStopReason.eosToken;
        if (onTokenEmitted != null) {
          onTokenEmitted(next, '', TokenOrigin.eos);
        }
        break;
      }

      var s = decode(next);
      fullText.write(s);

      if (onTokenEmitted != null) {
        onTokenEmitted(next, s, TokenOrigin.generated);
      }
      output.write(s);

      generatedTokens++;
      seen.increment(next);

      forward(next);
    }

    if (stopReason == TokenGenerationStopReason.maxTokensReached &&
        onTokenEmitted != null) {
      onTokenEmitted(0, '', TokenOrigin.maxTokensReached);
    }

    genWatch.stop();
    final genDuration = genWatch.elapsed;

    final promptTokens = toks.length;
    final totalTokens = promptTokens + generatedTokens;

    final promptSecs = promptDuration.inMicroseconds / 1000000.0;
    final genSecs = genDuration.inMicroseconds / 1000000.0;

    final promptTks = promptSecs == 0 ? 0.0 : promptTokens / promptSecs;
    final genTks = genSecs == 0 ? 0.0 : generatedTokens / genSecs;

    return TokenGenerationResult(
      prompt: prompt,
      output: output.toString(),
      seed: seed,
      random: random,
      maxTokens: maxTokens,
      temperature: temperature,
      repeatPenalty: repeatPenalty,
      promptTokens: promptTokens,
      generatedTokens: generatedTokens,
      totalTokens: totalTokens,
      promptDuration: promptDuration,
      generationDuration: genDuration,
      totalDuration: promptDuration + genDuration,
      promptTokensPerSecond: promptTks,
      generatedTokensPerSecond: genTks,
      stopReason: stopReason,
    );
  }
}
