@Timeout(Duration(minutes: 5))
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:huggingface_downloader/huggingface_downloader.dart';
import 'package:path/path.dart' as path;
import 'package:smollm2/smollm2.dart';
import 'package:test/test.dart';

void main() {
  group('SmolLM2 full export and inference workflow', () {
    test('HuggingFaceTB/SmolLM2-135M-Instruct', () async {
      // ------------------------------------------------------------
      // Create isolated temporary workspace for the integration test
      // ------------------------------------------------------------
      final tempDir = await Directory.systemTemp.createTemp(
        'smollm2_135m_test_',
      );

      try {
        // ------------------------------------------------------------
        // Step 1: Download Hugging Face SmolLM2 checkpoint
        // ------------------------------------------------------------
        const repo = 'HuggingFaceTB/SmolLM2-135M-Instruct';

        final localRepoDir = Directory(path.join(tempDir.path, repo));

        print(
          'Downloading Hugging Face checkpoint: $repo -> ${localRepoDir.path}',
        );

        final downloader = HuggingFaceDownloader();

        var downloadedFiles = await downloader.downloadSnapshot(
          repoId: repo,
          localDir: localRepoDir,
          allowExtensions: ['.json', '.txt', '.model', '.safetensors', '.bin'],
          excludeExtensions: [
            '.onnx',
            '.gguf',
            '.h5',
            '.msgpack',
            '.tflite',
            '.pt',
            '.pth',
            '.ot',
            '.ckpt',
          ],
        );

        print('** Downloaded files:');
        for (var f in downloadedFiles) {
          print('  -- ${f.path}');
        }

        // ------------------------------------------------------------
        // Step 2: Export to native SMOL binary
        // ------------------------------------------------------------
        final outputBinFile = File(
          path.join(localRepoDir.path, 'smollm2-q16.bin'),
        );

        print('Exporting checkpoint to native SMOL Q16 binary...');

        await exportSmolLM2(
          configPath: path.join(localRepoDir.path, 'config.json'),
          tokenizerPath: path.join(localRepoDir.path, 'tokenizer.json'),
          modelPath: path.join(localRepoDir.path, 'model.safetensors'),
          outputPath: outputBinFile.path,
          quantType: QuantType.q16,
        );

        expect(outputBinFile.existsSync(), isTrue);

        print('---------------------------------------------------');
        print('Loading `SmolLM2`: ${outputBinFile.path} ...');

        // ------------------------------------------------------------
        // Step 3: Load model into Dart inference engine
        // ------------------------------------------------------------
        final smollm = SmolLM2();

        await smollm.load(outputBinFile.path);

        // ------------------------------------------------------------
        // Step 4: Run deterministic generation
        // ------------------------------------------------------------
        const prompt = 'The capital of France is';

        print('---------------------------------------------------');

        var tokensEmitted =
            <(int tokenId, String tokenText, TokenOrigin origin)>[];

        var tokensText = <String>[];
        var tokensOrigin = <String>[];

        void onTokenEmitted(int tokenId, String tokenText, TokenOrigin origin) {
          tokensEmitted.add((tokenId, tokenText, origin));
          tokensText.add(tokenText);
          tokensOrigin.add(origin.name);
          stdout.write(tokenText);
        }

        var result = await smollm.generate(
          prompt,
          maxTokens: 40,
          temperature: 0.8,
          repeatPenalty: 1.1,
          seed: 12345,
          onTokenEmitted: onTokenEmitted,
        );

        await stdout.flush();

        print('\n---------------------------------------------------');

        print('tokensText:');
        print(json.encode(tokensText));
        print('');
        print('tokensOrigin:');
        print(json.encode(tokensOrigin));
        print('');

        expect(result.prompt, equals(prompt));
        expect(result.maxTokens, equals(40));
        expect(result.temperature, equals(0.8));
        expect(result.repeatPenalty, equals(1.1));
        expect(result.seed, equals(12345));

        expect(
          result.stopReason,
          equals(TokenGenerationStopReason.maxTokensReached),
        );

        expect(
          result.output,
          equals(
            'The capital of France is Paris. Paris, the City of Light, '
            'is a major center for art and culture, with many iconic '
            'landmarks such as the Eiffel Tower, Notre-Dame Cathedral, '
            'and Louvre Museum.',
          ),
        );

        expect(
          tokensText,
          equals([
            "The",
            " capital",
            " of",
            " France",
            " is",
            " Paris",
            ".",
            " Paris",
            ",",
            " the",
            " City",
            " of",
            " Light",
            ",",
            " is",
            " a",
            " major",
            " center",
            " for",
            " art",
            " and",
            " culture",
            ",",
            " with",
            " many",
            " iconic",
            " landmarks",
            " such",
            " as",
            " the",
            " E",
            "iffel",
            " Tower",
            ",",
            " Notre",
            "-",
            "D",
            "ame",
            " Cathedral",
            ",",
            " and",
            " Lou",
            "vre",
            " Museum",
            ".",
            "",
          ]),
        );

        expect(
          tokensOrigin,
          equals([
            "prompt",
            "prompt",
            "prompt",
            "prompt",
            "prompt",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "generated",
            "maxTokensReached",
          ]),
        );

        print('=================================================');

        final smollmJittered = SmolLM2();

        var jitterRandom = Random(123);
        await smollmJittered.load(
          outputBinFile.path,
          jitterRandom: jitterRandom,
        );

        var result2 = await smollmJittered.generate(
          prompt,
          maxTokens: 40,
          temperature: 0.1,
          repeatPenalty: 1.1,
          seed: 12345,
          includePromptInOutput: false,
        );

        expect(
          result2.output,
          equals(
            ' Paris.\n'
            '- The capital of Italy is Rome.\n'
            '- The capital of Spain is Madrid.\n'
            '- The capital city in Japan (capital) is Tokyo, and the rest are located in other',
          ),
        );
      } finally {
        // ------------------------------------------------------------
        // Cleanup temporary files created by the test
        // ------------------------------------------------------------
        if (tempDir.existsSync()) {
          await tempDir.delete(recursive: true);
        }
      }
    });

    test('HuggingFaceTB/SmolLM2-360M-Instruct', () async {
      // ------------------------------------------------------------
      // Create isolated temporary workspace for the integration test
      // ------------------------------------------------------------
      final tempDir = await Directory.systemTemp.createTemp(
        'smollm2_360m_test_',
      );

      try {
        // ------------------------------------------------------------
        // Step 1: Download Hugging Face SmolLM2 checkpoint
        // ------------------------------------------------------------
        const repo = 'HuggingFaceTB/SmolLM2-360M-Instruct';

        final localRepoDir = Directory(path.join(tempDir.path, repo));

        print(
          'Downloading Hugging Face checkpoint: $repo -> ${localRepoDir.path}',
        );

        final downloader = HuggingFaceDownloader();

        var downloadedFiles = await downloader.downloadSnapshot(
          repoId: repo,
          localDir: localRepoDir,
          allowExtensions: ['.json', '.txt', '.model', '.safetensors', '.bin'],
          excludeExtensions: [
            '.onnx',
            '.gguf',
            '.h5',
            '.msgpack',
            '.tflite',
            '.pt',
            '.pth',
            '.ot',
            '.ckpt',
          ],
        );

        print('** Downloaded files:');
        for (var f in downloadedFiles) {
          print('  -- ${f.path}');
        }

        // ------------------------------------------------------------
        // Step 2: Export to native SMOL binary
        // ------------------------------------------------------------
        final outputBinFile = File(
          path.join(localRepoDir.path, 'smollm2-q16.bin'),
        );

        print('Exporting checkpoint to native SMOL BF16 binary...');

        await exportSmolLM2(
          configPath: path.join(localRepoDir.path, 'config.json'),
          tokenizerPath: path.join(localRepoDir.path, 'tokenizer.json'),
          modelPath: path.join(localRepoDir.path, 'model.safetensors'),
          outputPath: outputBinFile.path,
          quantType: QuantType.bf16,
        );

        expect(outputBinFile.existsSync(), isTrue);

        print('---------------------------------------------------');
        print('Loading `SmolLM2`: ${outputBinFile.path} ...');

        // ------------------------------------------------------------
        // Step 3: Load model into Dart inference engine
        // ------------------------------------------------------------
        final smollm = SmolLM2();

        await smollm.load(outputBinFile.path);

        // ------------------------------------------------------------
        // Step 4: Run deterministic generation
        // ------------------------------------------------------------

        var promptRs = '''<|im_start|>system
You are a helpful AI assistant<|im_end|>
<|im_start|>user
How many r's in Strawberry?<|im_end|>
<|im_start|>assistant
''';

        print('---------------------------------------------------');

        var result = await smollm.generate(
          promptRs,
          maxTokens: 40,
          temperature: 0.1,
          repeatPenalty: 1.0,
          seed: 12345,
          includePromptInOutput: false,
        );

        await stdout.flush();

        expect(
          result.output,
          equals('''There are 3 r's in the word "Strawberry."'''),
        );

        print('\n---------------------------------------------------');
      } finally {
        // ------------------------------------------------------------
        // Cleanup temporary files created by the test
        // ------------------------------------------------------------
        if (tempDir.existsSync()) {
          await tempDir.delete(recursive: true);
        }
      }
    });
  });
}

/*
This integration test validates the complete smollm2 workflow in a temporary
isolated directory, leaving the project tree untouched.

Workflow:
1. create temporary directory
2. download HuggingFaceTB/SmolLM2-135M-Instruct
3. export checkpoint to native SMOL Q16 binary
4. load binary into SmolLM2
5. run deterministic generation
6. delete all temporary artifacts after test completion
*/
