import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  // Create a new SmolLM2 inference engine instance.
  final smollm = SmolLM2();

  // Load the exported SmolLM2 model into memory.
  // In this example we use the 135M Instruct quantized Q16 model.
  await smollm.load('models/smollm2-135m-instruct/smollm2-q16.bin');

  // Prompt to start the text generation.
  const prompt = 'The capital of France is';

  print('---------------------------------------------------');

  // Generate text directly to stdout using the configured sampling options:
  // - maxTokens: maximum number of tokens to generate
  // - temperature: controls randomness (lower = more deterministic)
  // - repeatPenalty: discourages repetitive output
  // - seed: ensures deterministic generation for reproducible results
  var output = await smollm.generate(
    prompt,
    maxTokens: 40,
    temperature: 0.8,
    repeatPenalty: 1.1,
    seed: 12345,
  );

  print('\n<<<\n$output\n>>>');
}

/*
Example output (maxTokens: 40, temperature: 0.8, repeatPenalty: 1.1, seed: 12345):

The capital of France is Paris. Paris is one of the most visited cities in
the world and is known for its history, architecture, gastronomy, and famous
monuments such as the Eiffel Tower, the Louvre, and Notre-Dame Cathedral.
*/
