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
