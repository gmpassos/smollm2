@Timeout(Duration(minutes: 5))
library;

import 'dart:io';

import 'package:huggingface_downloader/huggingface_downloader.dart';
import 'package:path/path.dart' as path;
import 'package:smollm2/smollm2.dart';
import 'package:test/test.dart';

void main() {
  test('SmolLM2 full export and inference workflow', () async {
    // ------------------------------------------------------------
    // Create isolated temporary workspace for the integration test
    // ------------------------------------------------------------
    final tempDir = await Directory.systemTemp.createTemp('smollm2_test_');

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

      print('Exporting checkpoint to native SMOL binary...');

      await exportSmolLM2(
        configPath: path.join(localRepoDir.path, 'config.json'),
        tokenizerPath: path.join(localRepoDir.path, 'tokenizer.json'),
        modelPath: path.join(localRepoDir.path, 'model.safetensors'),
        outputPath: outputBinFile.path,
        quantType: QuantType.q16,
      );

      expect(outputBinFile.existsSync(), isTrue);

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

      var output = await smollm.generate(
        prompt,
        maxTokens: 40,
        temperature: 0.8,
        repeatPenalty: 1.1,
        seed: 12345,
      );

      expect(
        output,
        equals(
          'The capital of France is Paris. Paris, the City of Light, '
          'is a major center for art and culture, with many iconic '
          'landmarks such as the Eiffel Tower, Notre-Dame Cathedral, '
          'and Louvre Museum.',
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
