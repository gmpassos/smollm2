import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'activations.dart';
import 'config.dart';
import 'data.dart';
import 'kv_cache.dart';
import 'quant_type.dart';
import 'token_generator.dart';
import 'tokenizer.dart';
import 'weights.dart';

class SmolLM2 implements TokenGenerator {
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

  Future<void> load(String modelPath) async {
    final modelFile = await File(modelPath).open();
    final dataReader = DataReader(modelFile);

    final quantType = _loadHeader(dataReader);

    _loadConfig(dataReader, quantType);
    _loadTokenizer(dataReader);
    _loadWeights(dataReader);

    await modelFile.close();

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

    print('** Model loaded');
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
      if (quantType != QuantType.q8 && quantType != QuantType.q16) {
        throw Exception('Q8 and Q16 only');
      }
      return quantType;
    }

    throw StateError('Bad version: $ver');
  }

  void _loadConfig(DataReader dataReader, QuantType quantType) {
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

    print(config);
  }

  void _loadTokenizer(DataReader dataReader) {
    final vocabLength = dataReader.readU32();
    final mergesLength = dataReader.readU32();

    final vocab = List.generate(vocabLength, (_) => dataReader.readString());
    final merges = List.generate(mergesLength, (_) {
      var a = dataReader.readString();
      var b = dataReader.readString();
      return (a, b);
    });

    tokenizer = Tokenizer(vocab: vocab, merges: merges);
    print(tokenizer);
  }

  void _loadWeights(DataReader dataReader) {
    weights = ModelWeights(config);
    weights.load(config, dataReader);
    print(weights);
  }

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

        ropeCos[p * hd + i] = c.toDouble();
        ropeCos[p * hd + hd ~/ 2 + i] = c.toDouble();

        ropeSin[p * hd + i] = s.toDouble();
        ropeSin[p * hd + hd ~/ 2 + i] = s.toDouble();
      }
    }
  }

  void resetCache() {
    for (final kv in kvCaches) {
      kv.cacheLen = 0;
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
    final outList = out.list;
    final outListX4 = out.listX4;

    final xList = x.list;
    final xListX4 = x.listX4;

    final wList = w.list;
    final wList4 = w.listX4;

    final n4 = n >> 2;

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

    //ssVec.reciprocal()

    double ss = ssVec.x + ssVec.y + ssVec.z + ssVec.w;

    // tail (if n not multiple of 4)
    for (int i = n4 << 2; i < n; i++) {
      final v = xList[i];
      ss += v * v;
    }

    ss = 1.0 / math.sqrt(ss / n + eps);

    final scale = ss;
    final scaleVec = Float32x4.splat(scale);

    final n4w = n4;

    // -------------------------
    // normalize (SIMD)
    // -------------------------
    for (int i = 0; i < n4w; i++) {
      outListX4[i] = xListX4[i] * scaleVec * wList4[i];
    }

    // -------------------------
    // tail (scalar)
    // -------------------------
    for (int i = n4w << 2; i < n; i++) {
      outList[i] = xList[i] * scale * wList[i];
    }
  }

  void applyRope(
    Float32List v,
    int vOffset,
    int hd,
    Float32List ropeCos,
    Float32List ropeSin,
    int ropeOffset,
  ) {
    final h = hd ~/ 2;

    for (int i = 0; i < h; i++) {
      final v0 = v[vOffset + i];
      final v1 = v[vOffset + i + h];

      final c0 = ropeCos[ropeOffset + i];
      final s0 = ropeSin[ropeOffset + i];
      final c1 = ropeCos[ropeOffset + i + h];
      final s1 = ropeSin[ropeOffset + i + h];

      v[vOffset + i] = (v0 * c0 - v1 * s0).toDouble();
      v[vOffset + i + h] = (v1 * c1 + v0 * s1).toDouble();
    }
  }

  Float32List forward(final int tok, final int pos) {
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

    return logits;
  }

  List<int> tokenize(String text, int max) {
    final t = tokenizer;

    final enc = StringBuffer();
    var start = true;

    for (final rune in text.runes) {
      final ch = String.fromCharCode(rune);

      if (ch == ' ') {
        if (!start) {
          enc.write('\u0120');
        }
      } else if (ch == '\n') {
        enc.write('\u010A');
        start = true;
        continue;
      } else {
        enc.write(ch);
        start = false;
      }
    }

    final encoded = enc.toString();
    final toks = <int>[];

    int p = 0;

    while (p < encoded.length && toks.length < max) {
      int bestLen = 0;
      int bestId = -1;

      for (int len = 1; len <= 32 && p + len <= encoded.length; len++) {
        final sub = encoded.substring(p, p + len);
        final id = t.findTok(sub);

        if (id >= 0) {
          bestLen = len;
          bestId = id;
        }
      }

      if (bestId >= 0) {
        toks.add(bestId);
        p += bestLen;
      } else {
        final id = t.findTok(encoded[p]);
        if (id >= 0) {
          toks.add(id);
        }
        p++;
      }
    }

    bool changed = true;
    int iter = 0;

    while (changed && toks.length > 1 && iter < 1000) {
      changed = false;
      iter++;

      for (final merge in t.mergePairs) {
        for (int j = 0; j < toks.length - 1; j++) {
          final t1 = t.vocab[toks[j]];
          final t2 = t.vocab[toks[j + 1]];

          if (t1 == merge.a && t2 == merge.b) {
            final merged = t1 + t2;
            final mid = t.findTok(merged);

            if (mid >= 0) {
              toks[j] = mid;
              toks.removeAt(j + 1);
              changed = true;
              break;
            }
          }
        }

        if (changed) {
          break;
        }
      }
    }

    return toks;
  }

  String decode(int tok) {
    if (tok < 0 || tok >= tokenizer.vocabSize) {
      return '';
    }

    final raw = tokenizer.vocab[tok];
    final out = StringBuffer();

    for (int i = 0; i < raw.length; i++) {
      final ch = raw[i];

      if (ch == '\u0120') {
        out.write(' ');
      } else if (ch == '\u010A') {
        out.write('\n');
      } else {
        out.write(ch);
      }
    }

    return out.toString();
  }

  int sample(
    Float32List logits,
    int vocab,
    double temperature,
    double repeatPenalty,
    Map<int, int> seen,
    math.Random rand,
  ) {
    // Apply repeat penalty:
    for (int i = 0; i < vocab; i++) {
      final count = seen[i] ?? 0;
      if (count > 0) {
        final factor = math.pow(repeatPenalty, count).toDouble();
        if (logits[i] > 0) {
          logits[i] /= factor;
        } else {
          logits[i] *= factor;
        }
      }
    }

    if (temperature <= 0) {
      int best = 0;
      for (int i = 1; i < vocab; i++) {
        if (logits[i] > logits[best]) {
          best = i;
        }
      }
      return best;
    }

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

  static final _randomSecure = math.Random.secure();

  static int generateSeed() {
    return _randomSecure.nextInt(0x7fffffff);
  }

  @override
  Future<TokenGenerationResult> generate(
    String prompt, {
    int maxTokens = TokenGenerator.defaultMaxTokens,
    double temperature = TokenGenerator.defaultTemperature,
    double repeatPenalty = TokenGenerator.defaultRepeatPenalty,
    int? seed,
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

    final toks = tokenize(prompt, 512);
    if (toks.isEmpty) {
      throw StateError('Tokenize failed');
    }

    resetCache();

    final seen = <int, int>{};

    Float32List? logits;

    // Prompt ingestion timing (optional separate metric)
    final promptStart = Stopwatch()..start();

    for (int i = 0; i < toks.length; i++) {
      final tok = toks[i];
      logits = forward(tok, i);
      seen.increment(tok);

      if (onTokenEmitted != null) {
        var s = decode(tok);
        onTokenEmitted(tok, s, TokenOrigin.prompt);
      }
    }

    promptStart.stop();
    final promptDuration = promptStart.elapsed;

    var stopReason = TokenGenerationStopReason.maxTokensReached;
    int generatedTokens = 0;
    final genWatch = Stopwatch()..start();

    for (int i = 0; i < maxTokens; i++) {
      final next = sample(
        logits!,
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

      if (onTokenEmitted != null) {
        onTokenEmitted(next, s, TokenOrigin.generated);
      }
      output.write(s);

      generatedTokens++;
      seen.increment(next);

      logits = forward(next, toks.length + i);
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
