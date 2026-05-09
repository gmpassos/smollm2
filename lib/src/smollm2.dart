import 'llm_token_generator.dart';
import 'runtime.dart';

/// SmolLM2 model loader and runtime interface.
///
/// Responsible for loading a quantized model (Q16) and preparing it
/// for inference in FP32, optionally applying controlled jitter during
/// dequantization to improve numerical robustness or variability.
class SmolLM2 extends LLMTokenGenerator {
  final LoggerFunction? logger;

  SmolLM2({this.logger}) : super(LLMRuntime(logger: logger));

  void log(Object? o) {
    final logger = this.logger;
    if (logger != null) {
      logger(o);
    }
  }

  List<int> tokenize(String text, int max) => runtime.tokenize(text, max);

  String decode(int tok) => runtime.decode(tok);

  static int generateSeed() => LLMTokenGenerator.generateSeed();
}
