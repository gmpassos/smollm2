import 'package:smollm2/smollm2.dart';

Future<void> main(List<String> args) async {
  String model = 'models/smollm2-135m-instruct/smollm2-q16.bin';
  // String model = 'models/smollm2-360m-instruct/smollm2-q8.bin';

  String prompt = 'The capital of France is';
  int maxTokens = 60;
  double temperature = 0.0;
  double repeatPenalty = 1.09;
  int? seed;

  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '-m':
        model = args[++i];
        break;
      case '-p':
        prompt = args[++i];
        break;
      case '-n':
        maxTokens = int.parse(args[++i]);
        break;
      case '-t':
        temperature = double.parse(args[++i]);
        break;
      case '-r':
        repeatPenalty = double.parse(args[++i]);
        break;
      case '-s':
        seed = int.parse(args[++i]);
        break;
      case '-h':
        print(
          'Usage: dart run smollm2.dart [-m modelPath] [-n maxTokens] [-t temperature] [-r repeatPenalty] [-s seed] [-p prompt]',
        );
        return;
    }
  }

  seed ??= SmolLM2.generateSeed();

  final m = SmolLM2();

  print('[SmolLM2]');
  print(
    '{ maxTokens: $maxTokens ; temperature: $temperature ; repetitionPenalty: $repeatPenalty ; seed: $seed }',
  );

  print('Loading $model...');
  await m.load(model);

  print('---------------------------------------------------------');

  await m.generate(
    prompt,
    maxTokens: maxTokens,
    temperature: temperature,
    repeatPenalty: repeatPenalty,
    seed: seed,
  );

  // await Future.delayed(Duration(hours: 10));
}
