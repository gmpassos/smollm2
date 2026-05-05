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
