import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  // Create a new SmolLM2 inference engine instance.
  final smollm = SmolLM2(logger: (o) => print('»» $o'));

  // Load the exported SmolLM2 model into memory.
  // In this example we use the 360m Instruct BF16 model.
  await smollm.load('models/smollm2-360m-instruct/smollm2-bf16.bin');

  // Prompt to start the text generation.

  var prompt = '''<|im_start|>system
You are a helpful AI assistant<|im_end|>
<|im_start|>user
How many r's in Strawberry?<|im_end|>
<|im_start|>assistant
''';

  print('---------------------------------------------------');

  // Generate text directly to stdout using the configured sampling options:
  // - maxTokens: maximum number of tokens to generate
  // - temperature: controls randomness (lower = more deterministic)
  // - repeatPenalty: discourages repetitive output
  // - seed: ensures deterministic generation for reproducible results
  var output = await smollm.generate(
    prompt,
    maxTokens: 40,
    temperature: 0.1,
    repeatPenalty: 1.0,
    seed: 12345,
  );

  print('\n<<<\n$output\n>>>');
}

// OUTPUT:
/*
»» Loading model: models/smollm2-360m-instruct/smollm2-bf16.bin (725094812 bytes) ...
»» Loaded (1.053ms): Config{quantType: QuantType.bf16, groupSize: 0, hiddenSize: 960, intermediateSize: 2560, numLayers: 32, numHeads: 15, numKvHeads: 5, vocabSize: 49152, maxSeqLen: 8192, headDim: 64}
»» Loaded (1.143s): Tokenizer{vocabSize: 49152, numMerges: 48900}
»» Loading weights...
»» Loaded (15.163s): ModelWeights{embedTokens: FP32Tensor{size: 47185920, rows: 49152 cols: 960, data: 47185920}, layers: 32, finalNorm: FP32Tensor{size: 960, rows: 1 cols: 960, data: 960}}
»» Model fully loaded 🚀 (16.369s)
---------------------------------------------------

<<<
<|im_start|>system
You are a helpful AI assistant<|im_end|>
<|im_start|>user
How many r's in Strawberry?<|im_end|>
<|im_start|>assistant
There are 3 r's in the word "Strawberry."
>>>
*/
