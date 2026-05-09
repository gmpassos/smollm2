import 'package:smollm2/smollm2.dart';

Future<void> main(List<String> args) async {
  // In this example we use the 360m Instruct BF16 model.
  var modelPath = args.isNotEmpty
      ? args[0]
      : 'models/smollm2-360m-instruct/smollm2-bf16.bin';

  // Create a new SmolLM2 inference engine instance.
  final smollm = SmolLM2(logger: (o) => print('»» $o'));

  // Load the exported SmolLM2 model into memory.
  await smollm.load(modelPath);

  // Prompt to start the text generation.
  var prompt = 'The capital of France is';

  print('---------------------------------------------------');

  // Generate text directly to stdout using the configured sampling options:
  // - maxTokens: maximum number of tokens to generate
  // - temperature: controls randomness (lower = more deterministic)
  // - repeatPenalty: discourages repetitive output
  // - seed: ensures deterministic generation for reproducible results
  var result = await smollm.generate(
    prompt,
    maxTokens: 60,
    temperature: 0.2,
    repeatPenalty: 1.1,
    seed: 123456, // not used for `temperature: 0.0`
  );

  // The token generation result.output:
  print('\n<<<\n${result.output}\n>>>');

  // The actual steam of text processed by the LLM:
  print('\n<<<\n${smollm.fullText}\n>>>');
}

// OUTPUT:
/*
»» Loading model: models/smollm2-360m-instruct/smollm2-bf16.bin (725094829 bytes) ...
»» Loaded (893µs): Config{quantType: QuantType.bf16, groupSize: 0, hiddenSize: 960, intermediateSize: 2560, numLayers: 32, numHeads: 15, numKvHeads: 5, vocabSize: 49152, maxSeqLen: 8192, headDim: 64}
»» Loaded (797.989ms): Tokenizer{vocabSize: 49152, numMerges: 48900, eosTokenIDs: [0, 2]}
»» Loading weights...
»» Loaded (6.870s): ModelWeights{embedTokens: FP32Tensor{size: 47185920, rows: 49152 cols: 960, data: 47185920}, layers: 32, finalNorm: FP32Tensor{size: 960, rows: 1 cols: 960, data: 960}}
»» Model fully loaded 🚀 (7.719s)
---------------------------------------------------

<<<
The capital of France is Paris.
Paris is the largest city in France.
Paris has a rich history and culture, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
>>>

<<<
The capital of France is Paris.
Paris is the largest city in France.
Paris has a rich history and culture, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.<|im_end|>
>>>
*/
