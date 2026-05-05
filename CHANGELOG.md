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
