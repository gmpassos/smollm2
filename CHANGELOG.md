## 1.0.6

- `bin/smollm2.dart`:
  - Added command line options `-j`, `-js`, `-jc` to enable jitter, set jitter seed, and jitter scale for model loading.
  - Updated model loading to pass jitter parameters to `SmolLM2.load`.
  - Added jitter-related parameters to startup logs.

- `bin/export_smollm2.dart`:
  - Added support for `-BF16` quantization flag.

- `lib/src/data.dart`:
  - Added `readU16` and `writeU16` methods to `DataReader` and `DataWriter`.
  - Added read/write byte count tracking fields.
  - Improved hashing extensions with two-hash variant and bit rotations.
  - Added `RandomExtension` with methods for generating jittered floats and noise.
  - Added `Hash64` class for incremental 64-bit hashing.
  - Added `DurationFormatting` extension for human-readable and seconds formatting.

- `lib/src/kv_cache.dart`:
  - Added `offset` method to `KVCache` for computing buffer offsets.

- `lib/src/quant_type.dart`:
  - Added new quant types: `q16PerBlock`, `bf16`.
  - Updated factory to support new quant types.

- `lib/src/smollm2.dart`:
  - Added detailed documentation for `SmolLM2` class and `load` method.
  - Added support for optional jitter during dequantization in model loading.
  - Added jitter parameters (`jitterSeed`, `jitterRandom`, `jitterScale`) to `load`.
  - Added timing and logging for loading phases (header, config, tokenizer, weights).
  - Updated `_loadWeights` to accept jitter parameters and pass them to weights loader.
  - Added `_loaded` flag and `isLoaded` getter.
  - Refactored model loading to support jitter injection during dequantization.
  - Improved `rmsNorm` and `applyRope` implementations.
  - Updated `forward` method to fix KV cache writes and optimize attention computation.
  - Updated `tokenize` and `decode` to use new `TokenizerEngine`.
  - Updated `sample` method to support eager greedy sampling and improved repeat penalty logic.

- `lib/src/tensor.dart`:
  - Introduced abstract `QTensor` base class for quantized tensors with multiple dequantization methods.
  - Added jittered and adaptive dequantization methods with optional stochastic jitter.
  - Updated `Q8Tensor` and `Q16Tensor` to support jittered dequantization via `toFP32Tensor` method.
  - Added default jitter scale constants for Q8 and Q16 tensors.
  - Added new `BF16Tensor` class with FP32 conversion and dot product support.

- `lib/src/token_generator.dart`:
  - Updated `TokenGenerationResult.statsSummary` to use new duration formatting extensions for human-readable timing.

- `lib/src/tokenizer.dart`:
  - Added support for added/special tokens in tokenizer.
  - Refactored tokenizer to use `TokenizerEngine` for tokenization and decoding.
  - `TokenizerEngine`:
    - Supports matching added tokens first (longest match).
    - Handles space and newline tokens explicitly.
    - Applies BPE merges using merge rank map.
    - Decodes tokens with whitespace and newline replacements.

- `lib/src/weights.dart`:
  - Updated `ModelWeights.load` and internal loading methods to accept jitter parameters.
  - Passed jitter parameters through to tensor reading methods.
  - Updated `_readQ8`, `_readQ16`, to support jittered FP32 tensor conversion.
  - added `_readBF16`.
  - Added support for BF16 quantization type.

- `pubspec.yaml`:
  - Updated dev dependencies:
    - `lints` from ^6.0.0 to ^6.1.0
    - `test` from ^1.25.6 to ^1.31.1
    - `huggingface_downloader` from ^1.0.0 to ^1.0.1
    - `path` from ^1.9.0 to ^1.9.1

- Added new example files:
  - `example/smollm2_completion_example.dart`: Basic text completion example using 360m BF16 model.
  - `example/smollm2_chat_example.dart`: Interactive multi-turn chat session example updated to use 360m BF16 model.
  - `example/smollm2_rs_in_strawberry_example.dart`: Prompt formatting and reasoning example using 360m BF16 model.
  - Added `example/example.md` with detailed usage instructions and example code snippets.
  - Removed deprecated example file `example/smollm2_example.dart`.

- Tests:
  - Added `test/smollm2_vocab.dart` containing encoded tokenizer vocabulary and merges.
  - Added `test/tokenizer_test.dart` with comprehensive tokenizer unit tests covering added tokens, BPE merges, and chat template encoding.
  - Updated `test/smollm2_test.dart` integration tests:
    - Added tests for 135M and 360M models including download, export, load, and deterministic generation.
    - Added test coverage for jittered model loading and generation.
    - Cleaned up temporary directories after tests.

## 1.0.5

- Documentation (`README.md`):
  - Added detailed TL;DR section for quick start with local LLM chat.
  - Added instructions for installing Dart SDK, Hugging Face model downloader CLI, and SmolLM2 CLI.
  - Added recommended commands to download small and larger SmolLM2 models.
  - Added instructions to export Hugging Face checkpoints to SMOL Q16 format.
  - Added example commands to run interactive chat with exported models.
  - Updated CLI usage examples to use global `smollm2` and `export_smollm2` commands instead of `dart run`.
  - Clarified installation instructions for adding `smollm2` dependency and global activation.
  - Improved formatting and consistency in CLI options and example usage.

- `bin/smollm2.dart`:
  - `_chatSession`: added `seed` parameter to `generate` call to support deterministic generation in chat mode.

- `lib/src/smollm2.dart` (`SmolLM2`):
  - In `generate` method, moved `resetCache()` call before initializing `_fullText` and `_seen` caches to ensure proper cache reset.

- `lib/src/token_generator.dart`:
  - Updated default repeat penalty for chat sessions from `1.02` to `1.0` for less penalization of repeated tokens during chat.

## 1.0.4

- Added chat mode support with interactive prompt-response loop in `bin/smollm2.dart`.
- `bin/smollm2.dart`:
  - Added command line options `-c` for chat mode and `-nc`/`--no-colored` to disable colored output.
  - Added colored output for tokens with distinct colors for prompt, generated tokens, EOS, and max tokens reached.
  - Added `_chatSession` function for interactive chat with system, user, and assistant roles.
  - Added `_promptComplete` function for single prompt completion with optional colored output.
- `lib/src/chat.dart`:
  - Added `ChatSession` and `ChatMessage` classes to manage chat history and build formatted prompts.
  - `ChatSession` enhancements:
    - Added optional `seed` parameter and internal `random` generator for deterministic sampling.
    - Added static `generateSeed()` method for secure random seed generation.
    - Added configurable chat template tokens `imStart` and `imEnd` with defaults `<|im_start|>` and `<|im_end|>`.
    - Updated `buildPrompt` to use configurable tokens and append assistant prompt.
    - Added `endsWithImEndToken` method to check if a response ends with the termination token.
- `lib/src/smollm2.dart`:
  - Added optional `logger` callback to `SmolLM2` for logging model loading and status messages.
  - Added detailed logging during model loading steps.
  - Changed `forward` method to track total and context tokens internally.
  - Added `totalTokens` and `contextTokens` getters to track tokens processed and cached.
  - Added `resetCache` method to reset KV caches and token counters.
  - Added incremental prompt ingestion with `ingest` method supporting partial prompt feeding and token emission.
  - Refactored `generate` method to use incremental prompt ingestion and track full generated text.
  - Added internal `_fullText` buffer to accumulate all decoded tokens.
  - Added internal `_seen` map to track token repetition counts across prompt and generation.
  - Updated `sample` method to use internal logits and repeat penalty logic.
- `lib/src/token_generator.dart`:
  - Added `isTerminal` property to `TokenOrigin` enum to identify terminal token emission events.
  - Added `random` field to `TokenGenerationResult` to expose RNG used during sampling.
  - Added default chat-specific temperature and repeat penalty constants.
  - Added `emmitPromptTokens` parameter to `generate` method to control prompt token emission callbacks.
- `lib/smollm2.dart`:
  - Exported new `chat.dart` module for chat session support.
- `example/smollm2_example.dart`:
  - Added logger callback to example `SmolLM2` instance for verbose output.
- Example:
  - Added `example/smollm2_chat_example.dart` demonstrating interactive chat session usage with token streaming, seed control, and proper prompt management.

## 1.0.3

- Added streaming token emission support to `SmolLM2.generate`:
  - Added `onTokenEmitted` callback parameter to receive tokens as they are generated.
  - Emitted tokens during prompt ingestion and generation with associated `TokenOrigin`.
  - Emitted special terminal tokens for EOS and max tokens reached.
- Introduced `TokenGenerator` interface and related types in `token_generator.dart`:
  - `TokenOrigin` enum to identify token source (prompt, generated, eos, maxTokensReached).
  - `OnTokenEmitted` callback typedef for streaming tokens.
  - `TokenGenerationStopReason` enum for generation stop reasons.
  - `TokenGenerationResult` class encapsulating generation output, parameters, token counts, timings, throughput, and stop reason.
  - `TokenGenerator` abstract class defining the `generate` method contract.
- Updated `SmolLM2` to implement `TokenGenerator`:
  - `generate` now returns `Future<TokenGenerationResult>` instead of raw string.
  - Added detailed timing and throughput measurements.
  - Supports streaming tokens via `onTokenEmitted`.
- Updated example CLI (`bin/smollm2.dart`) to:
  - Use `onTokenEmitted` callback to print tokens as they are generated.
  - Print generation statistics summary after completion.
- Added comprehensive integration test in `smollm2_test.dart`:
  - Tests full export, load, and deterministic generation workflow.
  - Captures and verifies emitted tokens and their origins.
  - Validates `TokenGenerationResult` fields and stop reason.
  - Prints emitted tokens and origins for inspection.

## 1.0.2

- `HFTokenizer`:
  - Updated `merges` field type to `List<(String, String)>`.
  - Improved `load` method to parse `merges` entries from either list pairs or space-separated strings.
- `TensorRepositoryLoader`:
  - Enhanced shard index detection to check multiple possible index file names (`.safetensors.index.json` and `.index.json`).
- `SmolLM2Exporter`:
  - Updated tokenizer merges serialization to write each merge as two separate strings.
- `SmolLM2`:
  - Updated tokenizer merges deserialization to read pairs of strings instead of single strings.
- `Tokenizer`:
  - Updated `merges` field type to `List<(String, String)>`.
  - Updated `_buildMergePairs` to use tuple elements directly instead of parsing strings.

## 1.0.1

- `pubspec.yaml`:
  - Updated SDK constraint from `^3.10.9` to `^3.10.0`.
  - Added `executables` section with `smollm2` and `export_smollm2`.

## 1.0.0

- Initial version.
