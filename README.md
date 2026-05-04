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
* 📦 Hugging Face safetensors exporter
* 🖥 CLI runner included
* 🔧 Programmatic API for Dart apps

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
````

or install with:

```bash
dart pub add smollm2
```

---

## Model Export

Before inference, a Hugging Face SmolLM2 model checkpoint must be converted into the native custom optimized `.bin`
format.

### Directory Mode

```bash
dart run bin/export_smollm2.dart -Q8 models/smollm2-135m-instruct/
```

or

```bash
dart run bin/export_smollm2.dart -Q16 models/smollm2-135m-instruct/
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
dart run bin/export_smollm2.dart \
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

Run a local generation:

```bash
dart run bin/smollm2.dart
```

Default parameters:

```text
model           = models/smollm2-135m-instruct/smollm2-q16.bin
prompt          = The capital of France is
maxTokens       = 60
temperature     = 0.0
repeatPenalty   = 1.09
seed            = auto-generated
```

---

### CLI Options

```bash
dart run bin/smollm2.dart [options]
```

| Option | Description              |
|--------|--------------------------|
| `-m`   | Model `.bin` file path   |
| `-p`   | Prompt text              |
| `-n`   | Maximum generated tokens |
| `-t`   | Temperature              |
| `-r`   | Repetition penalty       |
| `-s`   | Seed                     |
| `-h`   | Help                     |

---

### Example

```bash
dart run bin/smollm2.dart \
  -m models/smollm2-360m-instruct/smollm2-q8.bin \
  -p "Explain what quantum computing is in simple terms." \
  -n 120 \
  -t 0.7 \
  -r 1.12 \
  -s 12345
```

---

## Programmatic Usage

```dart
import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  final model = SmolLM2();

  await model.load('models/smollm2-135m-instruct/smollm2-q16.bin');

  await model.generate(
    'Write a short poem about the sea.',
    maxTokens: 80,
    temperature: 0.8,
    repeatPenalty: 1.10,
    seed: 42,
  );
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

## Runtime Statistics

After generation the engine reports:

* prompt token count
* generated token count
* total tokens
* prompt ingestion speed
* generation speed

Example:

```text
--- stats ---
prompt tokens    : 6
generated tokens : 60
total tokens     : 66
prompt ingest    : 0.842 s (7.12 tk/s)
generation       : 5.101 s (11.76 tk/s)
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

* portability
* simplicity
* pure Dart experimentation
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
