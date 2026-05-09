import 'dart:math' as math;

import 'config.dart';
import 'data.dart';
import 'runtime.dart';
import 'token_generator.dart';
import 'tokenizer.dart';

/// Token generator implementation backed by an [LLMRuntime].
///
/// This class manages prompt ingestion, token generation, KV cache state,
/// and token decoding for autoregressive language model inference.
class LLMTokenGenerator extends TokenGenerator {
  /// Runtime used to execute model inference.
  final LLMRuntime runtime;

  /// Creates a token generator using the provided [runtime].
  ///
  /// - [eosTokenId]: Token ID treated as the End-Of-Sequence marker during
  ///   generation.
  LLMTokenGenerator(this.runtime);

  static final _randomSecure = math.Random.secure();

  static int generateSeed() {
    return _randomSecure.nextInt(0x7fffffff);
  }

  Config get config => runtime.config;

  Tokenizer get tokenizer => runtime.tokenizer;

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

    return runtime.load(
      modelPath,
      applyDequantizationJitter: applyDequantizationJitter,
      jitterSeed: jitterSeed,
      jitterRandom: jitterRandom,
      jitterScale: jitterScale,
    );
  }

  /// Resets the underlying runtime KV caches and clears generation state.
  ///
  /// After calling this method, subsequent inference starts with a fresh
  /// context and no previously cached attention state.
  ///
  /// See [LLMRuntime.resetCache].
  void resetCache() {
    runtime.resetCache();
  }

  /// Total number of tokens processed by this model instance since the last
  /// [resetCache] call.
  ///
  /// This counter is cumulative and monotonic: every ingested prompt token and
  /// every generated token increments this value.
  ///
  /// Unlike [contextTokens], this value is not limited by the active context
  /// window and continues growing for the full lifetime of the inference session.
  int get totalTokens => runtime.totalTokens;

  StringBuffer? _fullText;

  String get fullText => _fullText?.toString() ?? '';

  int get fullTextLength => _fullText?.length ?? 0;

  Map<int, int>? _seen;

  /// Resets the chat/session state.
  ///
  /// This clears:
  /// - All KV caches and cached attention state.
  /// - Previously seen/generated content tracking.
  /// - The accumulated full generated text.
  ///
  /// After calling this method, the session behaves as a fresh interaction.
  void reset() {
    resetCache();
    _seen = null;
    _fullText = null;
  }

  /// Ingests a prompt into the model context without generating new tokens.
  ///
  /// The prompt is tokenized and forwarded through the model so its attention
  /// state becomes part of the active KV cache. This is useful for preparing
  /// the context before generation.
  ///
  /// Returns:
  /// - [toks]: The tokenized representation of the ingested prompt.
  /// - [promptDuration]: Total time spent ingesting the prompt tokens.
  ///
  /// - [prompt]: Input text to tokenize and ingest.
  ///
  /// - [emmitPromptTokens]: Whether prompt tokens should trigger
  ///   [onTokenEmitted] callbacks.
  ///
  /// - [onTokenEmitted]: Optional callback invoked for each emitted prompt
  ///   token during ingestion.
  Future<({List<int> toks, Duration promptDuration})> ingest(
    String prompt, {
    bool emmitPromptTokens = true,
    OnTokenEmitted? onTokenEmitted,
  }) async {
    final toks = runtime.tokenize(prompt, 512);
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

    final tokenizer = runtime.tokenizer;

    // Prompt ingestion timing (optional separate metric)
    final promptStart = Stopwatch()..start();

    for (int i = 0; i < toks.length; i++) {
      final tok = toks[i];
      runtime.forward(tok);

      final specialTok = tokenizer.isSpecialTok(tok);
      if (!specialTok) {
        // Track repetition only for semantic tokens (ignore special/control tokens):
        seen.increment(tok);
      }

      var s = runtime.decode(tok);
      fullText.write(s);

      if (onTokenEmitted != null && emmitPromptTokens) {
        if (specialTok) {
          s = renderSpecialToken(s);
        }
        onTokenEmitted(tok, s, TokenOrigin.prompt);
      }
    }

    promptStart.stop();
    final promptDuration = promptStart.elapsed;

    return (toks: toks, promptDuration: promptDuration);
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

    final (toks: promptTokens, promptDuration: promptDuration) = await ingest(
      prompt,
      onTokenEmitted: onTokenEmitted,
      emmitPromptTokens: emmitPromptTokens,
    );

    final fullText = _fullText!;
    final seen = _seen!;

    var stopReason = TokenGenerationStopReason.maxTokensReached;

    final genWatch = Stopwatch()..start();

    final tokenizer = runtime.tokenizer;
    final generatedTokens = <int>[];

    for (int i = 0; i < maxTokens; i++) {
      final next = runtime.sample(
        config.vocabSize,
        temperature,
        repeatPenalty,
        seen,
        random,
      );

      runtime.forward(next);
      generatedTokens.add(next);

      var s = runtime.decode(next);
      fullText.write(s);

      final eosToken = tokenizer.isEOSTok(next);

      if (eosToken) {
        stopReason = TokenGenerationStopReason.eosToken;
        if (onTokenEmitted != null) {
          onTokenEmitted(next, '', TokenOrigin.eos);
        }
        break;
      } else {
        var specialTok = tokenizer.isSpecialTok(next);

        if (specialTok) {
          s = renderSpecialToken(s);
        } else {
          // Track repetition only for semantic tokens (ignore special/control tokens):
          seen.increment(next);
        }

        if (onTokenEmitted != null) {
          onTokenEmitted(next, s, TokenOrigin.generated);
        }
        output.write(s);
      }
    }

    if (stopReason == TokenGenerationStopReason.maxTokensReached &&
        onTokenEmitted != null) {
      onTokenEmitted(0, '', TokenOrigin.maxTokensReached);
    }

    genWatch.stop();
    final genDuration = genWatch.elapsed;

    final promptSecs = promptDuration.inMicroseconds / 1000000.0;
    final genSecs = genDuration.inMicroseconds / 1000000.0;

    final promptTks = promptSecs == 0 ? 0.0 : promptTokens.length / promptSecs;
    final genTks = genSecs == 0 ? 0.0 : generatedTokens.length / genSecs;

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
      promptDuration: promptDuration,
      generationDuration: genDuration,
      totalDuration: promptDuration + genDuration,
      promptTokensPerSecond: promptTks,
      generatedTokensPerSecond: genTks,
      stopReason: stopReason,
    );
  }

  String renderSpecialToken(String s) {
    switch (s) {
      case '<|endoftext|>':
        return '\n[EOF]\n';

      case '<|im_start|>':
        return '\n[';

      case '<|im_end|>':
        return ']\n';

      case '<repo_name>':
      case '<reponame>':
        return '\n[repo: ';

      case '<file_sep>':
        return '\n---- file separator ----\n';

      case '<filename>':
        return '\n[file: ';

      case '<gh_stars>':
        return ' stars=';

      case '<issue_start>':
        return '\n[issue start]\n';

      case '<issue_comment>':
        return '\n  > ';

      case '<issue_closed>':
        return '\n[issue closed]\n';

      case '<jupyter_start>':
        return '\n=== notebook ===\n';

      case '<jupyter_text>':
        return '\n[text]\n';

      case '<jupyter_code>':
        return '\n[code]\n';

      case '<jupyter_output>':
        return '\n[output]\n';

      case '<jupyter_script>':
        return '\n[script]\n';

      case '<empty_output>':
        return '\n[empty]\n';

      default:
        return s;
    }
  }
}
