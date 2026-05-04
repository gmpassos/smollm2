import 'dart:io';

import 'package:smollm2/smollm2.dart';

Future<void> main(List<String> args) async {
  // ----------------------------
  // Quant flag parsing
  // ----------------------------
  QuantType quantType = QuantType.q8; // default

  final filteredArgs = <String>[];

  for (final arg in args) {
    if (arg == '-Q8') {
      quantType = QuantType.q8;
      continue;
    }
    if (arg == '-Q16') {
      quantType = QuantType.q16;
      continue;
    }
    filteredArgs.add(arg);
  }

  // ----------------------------
  // Argument validation
  // ----------------------------
  if (filteredArgs.isEmpty ||
      filteredArgs.length == 3 ||
      filteredArgs.length > 4) {
    _printUsage();
    exit(1);
  }

  late final String configPath;
  late final String tokenizerPath;
  late final String modelPath;
  late final String outputPath;

  // ----------------------------
  // Directory mode
  // ----------------------------
  if (filteredArgs.length == 1 || filteredArgs.length == 2) {
    final dir = filteredArgs[0].replaceAll(RegExp(r'[/\\]+$'), '');

    configPath = '$dir/config.json';
    tokenizerPath = '$dir/tokenizer.json';

    final singleModel = '$dir/model.safetensors';
    final shardedModel = '$dir/model';

    if (File(singleModel).existsSync()) {
      modelPath = singleModel;
    } else {
      modelPath = shardedModel;
    }

    outputPath = filteredArgs.length == 2
        ? filteredArgs[1]
        : '$dir/smollm2-${quantType.name}.bin';
  }
  // ----------------------------
  // Explicit file mode
  // ----------------------------
  else if (filteredArgs.length == 4) {
    configPath = filteredArgs[0];
    tokenizerPath = filteredArgs[1];
    modelPath = filteredArgs[2];
    outputPath = filteredArgs[3];
  } else {
    _printUsage();
    exit(1);
  }

  try {
    final sw = Stopwatch()..start();

    await exportSmolLM2(
      configPath: configPath,
      tokenizerPath: tokenizerPath,
      modelPath: modelPath,
      outputPath: outputPath,
      quantType: quantType,
    );

    sw.stop();

    print('');
    print('QuantType: $quantType');
    print('Export completed in ${sw.elapsed.inSeconds}s');
    print('Output: $outputPath');
  } catch (e, st) {
    print('');
    print('Export failed: $e');
    print(st);
    exit(2);
  }
}

void _printUsage() {
  print('');
  print('SmolLM2 Exporter');
  print('');
  print('Usage:');
  print('  dart run bin/export_smollm2.dart -Q8  <model_dir>');
  print('  dart run bin/export_smollm2.dart -Q16 <model_dir>');
  print('');
  print('  dart run bin/export_smollm2.dart <model_dir> [output.bin]');
  print('');
  print(
    '  dart run bin/export_smollm2.dart <config.json> <tokenizer.json> <model.safetensors|model> <output.bin>',
  );
  print('');
  print('Directory mode expects:');
  print('  config.json');
  print('  tokenizer.json');
  print('  model.safetensors OR model.index.json + shards');
  print('');
  print('Examples:');
  print('  dart run bin/export_smollm2.dart -Q8 models/smollm2/');
  print('  dart run bin/export_smollm2.dart -Q16 models/smollm2/');
  print('');
}
