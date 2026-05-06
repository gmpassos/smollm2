# smollm2

[![pub package](https://img.shields.io/pub/v/smollm2.svg?logo=dart&logoColor=00b9fc)](https://pub.dev/packages/smollm2)
[![Null Safety](https://img.shields.io/badge/null-safety-brightgreen)](https://dart.dev/null-safety)
[![GitHub Tag](https://img.shields.io/github/v/tag/gmpassos/smollm2?logo=git&logoColor=white)](https://github.com/gmpassos/smollm2/releases)
[![Last Commit](https://img.shields.io/github/last-commit/gmpassos/smollm2?logo=github&logoColor=white)](https://github.com/gmpassos/smollm2/commits/master)
[![License](https://img.shields.io/github/license/gmpassos/smollm2?logo=open-source-initiative&logoColor=green)](https://github.com/gmpassos/smollm2/blob/master/LICENSE)

`smollm2` is a **pure Dart LLM inference engine, local language model runtime, and Hugging Face exporter for SmolLM2 language models**.

It allows you to:

* run **SmolLM2 text generation locally**
* export Hugging Face SmolLM2 checkpoints into an optimized Dart binary format
* use **Q8** or **Q16** quantized weights
* generate deterministic or seeded outputs
* embed inference directly inside Dart applications

No Python runtime, no llama.cpp dependency, and no external native bindings are required.

---

## Features

* 🧠 Pure Dart transformer inference
* ⚡ SIMD optimized math kernels
* 💾 Built-in Q8 and Q16 quantization formats
* 🔁 KV cache for autoregressive generation
* 🌀 RoPE positional embeddings
* 🎲 Temperature + repetition penalty + deterministic seed
- 💬 Chat mode with conversation memory
- 🖥 CLI tool included
* 🔧 Programmatic API for Dart apps

---
## TL;DR - I just want to chat with a local LLM

### If you don’t have Dart SDK yet

Go to: [https://dart.dev/get-dart](https://dart.dev/get-dart)

---

### Install required tools

**Install the Hugging Face model downloader CLI used to fetch SmolLM2 checkpoints:**

```bash
dart pub global activate huggingface_downloader
```

**Install the SmolLM2 CLI used for export and local inference (provides `smollm2` and `export_smollm2`):**

```bash
dart pub global activate smollm2
```

---

### Download a model (recommended options)

**Small model (fast, lightweight, good for testing)**

```bash
huggingface_downloader \
  HuggingFaceTB/SmolLM2-135M-Instruct \
  ./models/smollm2-135m \
  --llm-only
```

**Larger model (better quality, slower, more capable)**

```bash
huggingface_downloader \
  HuggingFaceTB/SmolLM2-360M-Instruct \
  ./models/smollm2-360m \
  --llm-only
```

---

### Export model to SMOL format (Q16)

**Convert Hugging Face checkpoint into a high-precision single binary (Q16) for better output quality. This generates a `*-q16.bin` file inside the folder:**

```bash
export_smollm2 -Q16 models/smollm2-135m/
```

(or)

```bash
export_smollm2 -Q16 models/smollm2-360m/
```

---

### Start chat

**Run the interactive local chat interface using the exported model (use the model you exported):**

**135M model**

```bash
smollm2 \
  -m models/smollm2-135m/smollm2-q16.bin \
  -c
```

**360M model**

```bash
smollm2 \
  -m models/smollm2-360m/smollm2-q16.bin \
  -c
```

---

**Enjoy your fully local LLM 🚀 — no servers 🌐, no APIs 🔌, just your machine running the model 💻 (in pure Dart 🎯)**

---

## Supported Models

This package is designed for the [**SmolLM2 family**](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
published by [**Hugging Face Smol Models Research**](https://huggingface.co/HuggingFaceTB).

Typical supported checkpoints:

* SmolLM2-135M-Instruct
* SmolLM2-360M-Instruct

Other SmolLM2 variants with the same architecture may also work.

---

## Installation

Add to `pubspec.yaml`:

```yaml
dependencies:
  smollm2: ^latest_version
```

or add with:

```bash
dart pub add smollm2
```

or activate the CLI globally:

```bash
dart pub global activate smollm2
```

---

## Model Export

Before inference, a Hugging Face SmolLM2 model checkpoint must be converted into the native custom optimized `.bin`
format.

### Directory Mode

```bash
export_smollm2 -Q8 models/smollm2-135m-instruct/
```

or:

```bash
export_smollm2 -Q16 models/smollm2-135m-instruct/
```

Expected directory contents:

```text
config.json
tokenizer.json
model.safetensors
```

or:

```text
config.json
tokenizer.json
model.index.json + shard files
```

The exporter automatically detects whether the model is single-file or sharded.

Generated output example:

```text
models/smollm2-135m-instruct/smollm2-q8.bin
```

---

### Explicit File Mode

```bash
export_smollm2 \
  config.json \
  tokenizer.json \
  model.safetensors \
smollm2-q8.bin
```

---

### Export Notes

Available quantization formats:

* `-Q8` → smaller file, faster loading
* `-Q16` → larger file, better numeric precision

The exporter converts:

* configuration
* tokenizer vocabulary
* merge pairs
* all transformer weights

into a single portable binary file optimized for Dart runtime loading.

### Custom Binary Format

The exporter writes the Hugging Face model checkpoint into a **single custom `SMOL` binary file** designed specifically
for fast Dart loading and low runtime overhead.

This binary format stores, in sequence:

* package header and format version
* quantization metadata
* model configuration
* tokenizer vocabulary
* tokenizer merge pairs
* all transformer tensors already converted to the selected quantized representation

Unlike Hugging Face safetensors, which require parsing many named tensors and JSON metadata at runtime, the `SMOL`
format is a **direct sequential memory layout**. This allows the Dart engine to read the file in one pass with minimal
allocations and without expensive tensor name resolution.

Additional advantages:

* faster startup time
* much lower parsing complexity
* portable single-file deployment
* deterministic tensor ordering
* direct compatibility with Q8/Q16 internal kernels

The file begins with the magic bytes `SMOL`, followed by a version field, making the format extensible for future
quantization modes and runtime improvements.

---

## CLI Inference

### CLI Options

```bash
smollm2 [options]
```

| Option | Description            |
| ------ | ---------------------- |
| -m     | model path             |
| -p     | prompt                 |
| -n     | max tokens             |
| -t     | temperature            |
| -r     | repetition penalty     |
| -s     | seed                   |
| -c     | chat mode              |
| -nc    | disable colored output |
| -h     | help                   |

### Text Completion Mode

Run SmolLM2 as a **text continuation model** using `-p`.

```bash
smollm2 \
  -m models/smollm2-135m-instruct/smollm2-q8.bin \
  -t 0.1 \
  -r 1.01 \
  -n 40 \
  -p "The capital of France is"
```

Example output:

```id="pvdx9a"
=== SmolLM2 ===
»» Parameters: maxTokens: 40 ; temperature: 0.1 ; repetitionPenalty: 1.01 ; seed: 1377160423 ; colored: true
 » Loading model: models/smollm2-135m-instruct/smollm2-q8.bin ...
 » Config{quantType: QuantType.q8, groupSize: 0, hiddenSize: 576, intermediateSize: 1536, numLayers: 30, numHeads: 9, numKvHeads: 3, vocabSize: 49152, maxSeqLen: 8192, headDim: 64}
 » Tokenizer{vocabSize: 49152, numMerges: 48900}
 » ModelWeights{embedTokens: FP32Tensor{size: 28311552, rows: 49152 cols: 576, data: 28311552}, layers: 30, finalNorm: FP32Tensor{size: 576, rows: 1 cols: 576, data: 576}}
 » Model loaded
---------------------------------------------------------
The capital of France is Paris, a vibrant city known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also home to several world-class museums, including¤
---------------------------------------------------------
=== Token Generation Stats ===
prompt.length    : 24
output.length    : 205

seed             : 1377160423
maxTokens        : 40
temperature      : 0.1
repeatPenalty    : 1.01

stop reason      : TokenGenerationStopReason.maxTokensReached

prompt tokens    : 5
generated tokens : 40
total tokens     : 45

prompt ingest    : 0.291 s (17.18 tk/s)
generation       : 1.219 s (32.81 tk/s)
total            : 1.510 s (29.80 tk/s)

```

Key behavior:

* `-p` provides a **prefix to be completed**
* Model continues the text naturally (no instruction format)
* Output is a pure continuation of the input string
* Stops when `maxTokens` is reached or `EOS` is triggered

---

### Chat Mode

Run SmolLM2 in interactive chat mode using `-c`.

```bash
smollm2 \
  -m models/smollm2-135m-instruct/smollm2-q8.bin \
  -t 0.1 \
  -r 1.01 \
  -c
```

Example session:

```text
=== SmolLM2 ===
»» Parameters: maxTokens: 200 ; temperature: 0.1 ; repetitionPenalty: 1.01 ; seed: 1687595747 ; colored: true
 » Loading model: models/smollm2-135m-instruct/smollm2-q8.bin ...
 » Config{quantType: QuantType.q8, groupSize: 0, hiddenSize: 576, intermediateSize: 1536, numLayers: 30, numHeads: 9, numKvHeads: 3, vocabSize: 49152, maxSeqLen: 8192, headDim: 64}
 » Tokenizer{vocabSize: 49152, numMerges: 48900}
 » ModelWeights{embedTokens: FP32Tensor{size: 28311552, rows: 49152 cols: 576, data: 28311552}, layers: 30, finalNorm: FP32Tensor{size: 576, rows: 1 cols: 576, data: 576}}
 » Model loaded
---------------------------------------------------------
Chat mode enabled. Type "exit" to quit.
---------------------------------------------------------

You › Hello

 AI › Hello! How can I help you today?

You › Who is Isaac Newton?

 AI › Isaac Newton was an English mathematician, physicist, and astronomer who made major contributions to classical physics.

You › exit
---------------------------------------------------------
Full processed text:
---------------------------------------------------------

<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hello! How can I help you today?<|im_end|>
<|im_start|>user
Who is Isaac Newton?<|im_end|>
<|im_start|>assistant
Isaac Newton was an English mathematician, physicist, and astronomer who made major contributions to classical physics.<|im_end|>

---------------------------------------------------------
```

Key behavior:

* Each user input is appended to chat history
* Model generates assistant responses turn by turn
* Full formatted context uses `<|im_start|>` / `<|im_end|>` chat template
* Typing `exit` ends the session and prints the full serialized prompt history

---

## Programmatic Usage

### Text generation

```dart
import 'dart:io';
import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  final model = SmolLM2();

  const modelPath =
      'models/smollm2-135m-instruct/smollm2-q16.bin';

  await model.load(modelPath);

  // This is a prefix to be completed
  const prefix = 'The sea was calm and';

  print('Prefix: $prefix');
  print('\n--- completion ---\n');

  final result = await model.generate(
    prefix,
    maxTokens: 80,
    temperature: 0.8,
    seed: 42,
    repeatPenalty: SmolLM2.defaultRepeatPenalty,
    onTokenEmitted: (token, text, origin) {
      stdout.write(text);
    },
  );

  print('\n\n--- stats ---');
  print(result.statsSummary());
}
```

---

### Chat API

```dart
import 'dart:io';
import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  final smollm = SmolLM2();
  await smollm.load('models/smollm2-135m-instruct/smollm2-q16.bin');

  final chat = ChatSession();
  chat.addSystem('You are a helpful assistant.');

  var messagesOffset = 0;

  void onTokenEmitted(int t, String s, TokenOrigin o) {
    stdout.write(s);
  }

  print('Chat ready. Type "exit" to quit.');

  while (true) {
    stdout.write('\nYou › ');
    final input = stdin.readLineSync();
    if (input == null) continue;

    if (input.trim().toLowerCase() == 'exit') break;

    chat.addUser(input);

    final prompt = chat.buildPrompt(offset: messagesOffset);

    stdout.write('AI › ');

    var result = await smollm.generate(
      prompt,
      includePromptInOutput: false,
      emmitPromptTokens: false,
      onTokenEmitted: onTokenEmitted,
    );

    final assistantText = result.output;
    chat.addAssistant(assistantText);

    messagesOffset = chat.length;

    stdout.write('\n');
  }
}
```

---

## Generation Parameters

### Temperature

Controls randomness.

| Value       | Behavior                     |
|-------------|------------------------------|
| `0.0`       | Fully deterministic / greedy |
| `0.2 - 0.5` | Conservative                 |
| `0.6 - 0.9` | Balanced                     |
| `1.0+`      | Highly creative / unstable   |

---

### Repetition Penalty

Discourages token loops and repeated phrases.

Typical values:

| Value         | Behavior       |
|---------------|----------------|
| `1.00`        | disabled       |
| `1.05 - 1.10` | light control  |
| `1.10 - 1.20` | strong control |

---

### Seed

Generation is reproducible when the same:

* prompt
* model
* temperature
* repetition penalty
* seed

are used together.

Random seed can also be generated automatically:

```dart
final seed = SmolLM2.generateSeed();
```

---

## TokenGenerationResult stats

Example of `TokenGenerationResult.statsSummary()`:

```text
=== SmolLM2 ===
»» Parameters: maxTokens: 40 ; temperature: 0.1 ; repetitionPenalty: 1.01 ; seed: 101836062 ; colored: true
 » Loading model: models/smollm2-135m-instruct/smollm2-q8.bin ...
 » Config{quantType: QuantType.q8, groupSize: 0, hiddenSize: 576, intermediateSize: 1536, numLayers: 30, numHeads: 9, numKvHeads: 3, vocabSize: 49152, maxSeqLen: 8192, headDim: 64}
 » Tokenizer{vocabSize: 49152, numMerges: 48900}
 » ModelWeights{embedTokens: FP32Tensor{size: 28311552, rows: 49152 cols: 576, data: 28311552}, layers: 30, finalNorm: FP32Tensor{size: 576, rows: 1 cols: 576, data: 576}}
 » Model loaded
---------------------------------------------------------
The capital of France is Paris, a city known for its historical landmarks, culture, and cultural institutions. Paris is a major center of commerce, finance, and education in the world.                                                                                                                                                                                      
                                                                                                                                                                                       
Paris is also a major center¤                                                                                                                                                          
---------------------------------------------------------
=== Token Generation Stats ===
prompt.length    : 24
output.length    : 214

seed             : 101836062
maxTokens        : 40
temperature      : 0.1
repeatPenalty    : 1.01

stop reason      : TokenGenerationStopReason.maxTokensReached

prompt tokens    : 5
generated tokens : 40
total tokens     : 45

prompt ingest    : 0.405 s (12.34 tk/s)
generation       : 1.262 s (31.69 tk/s)
total            : 1.667 s (26.99 tk/s)

```

---

## Downloading SmolLM2 Models from Hugging Face

SmolLM2 checkpoints can be downloaded directly from Hugging Face using the companion package [`huggingface_downloader`](https://pub.dev/packages/huggingface_downloader), a Dart CLI utility for resumable and structured model downloads.

This is especially useful because LLM checkpoints are large and may include multiple shard files.

Install globally:

```bash id="hd1"
dart pub global activate huggingface_downloader
```

---

### Download SmolLM2-135M-Instruct

```bash id="hd2"
huggingface_downloader \
  HuggingFaceTB/SmolLM2-135M-Instruct \
  ./models/smollm2-135m \
  --llm-only
```

---

### Download SmolLM2-360M-Instruct

```bash id="hd3"
huggingface_downloader \
  HuggingFaceTB/SmolLM2-360M-Instruct \
  ./models/smollm2-360m \
  --llm-only
```

---

### What `--llm-only` Does

The `--llm-only` flag downloads only the files required for language model export and inference, skipping unrelated repository assets such as README files, training metadata, images, or auxiliary resources.

Typical downloaded structure:

```text id="hd4"
models/smollm2-135m/HuggingFaceTB/SmolLM2-135M-Instruct/
├── config.json
├── tokenizer.json
├── model.safetensors
```

For sharded checkpoints:

```text id="hd5"
models/smollm2-360m/HuggingFaceTB/SmolLM2-360M-Instruct/
├── config.json
├── tokenizer.json
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
```

---

### Next Step After Download

After downloading, continue to [Model Export](#model-export) to convert the checkpoint into the optimized native `SMOL` binary format.

---

## Internal Architecture

`smollm2` implements the full SmolLM2 forward pass in Dart:

* tokenizer encoding + BPE merges
* embedding lookup
* RMSNorm
* QKV projections
* RoPE application
* grouped-query attention
* KV cache storage
* softmax attention
* SwiGLU MLP
* final projection to logits
* temperature/repetition sampling

Optimizations include:

* SIMD `Float32x4` vector math
* reusable activation buffers
* cached FP32 embedding matrix
* precomputed RoPE sin/cos tables
* quantized tensor loading

---

## Performance Goals

This project focuses on:

* pure Dart runtime
* portability
* simplicity
* educational transformer implementation
* local offline inference

It is not intended to outperform native CUDA/Metal inference engines, but aims to provide a lightweight and hackable LLM
runtime fully inside Dart.

---

## Future Improvements

Planned possible additions:

* top-k / top-p sampling
* chat template helpers
* streaming callback API
* CUDA/GPU accelerated tensor kernels
* Metal / Vulkan backend experimentation
* additional quantization modes
* isolate-based parallel tensor ops
* batched token generation
* conversational chat session helpers

---

## Issues & Feature Requests

Please report bugs or request features via the
[issue tracker][tracker].

[tracker]: https://github.com/gmpassos/smollm2/issues

---

## Author

Graciliano M. Passos: [gmpassos@GitHub][github]

[github]: https://github.com/gmpassos

---

## License

[Apache License - Version 2.0][apache_license]

[apache_license]: https://www.apache.org/licenses/LICENSE-2.0.txt
