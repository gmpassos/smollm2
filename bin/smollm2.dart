import 'dart:io';

import 'package:smollm2/smollm2.dart';

Future<void> main(List<String> args) async {
  String model = 'models/smollm2-135m-instruct/smollm2-q16.bin';

  int maxTokens = 200;
  double? temperature;
  double? repeatPenalty;

  int? seed;
  bool chatMode = false;
  bool colored = true;
  String? singlePrompt;

  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '-m':
        model = args[++i];
        break;
      case '-p':
        singlePrompt = args[++i];
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
      case '-nc':
      case '--no-colored':
        colored = false;
        break;
      case '-c':
        chatMode = true;
        break;
      case '-h':
        print(
          'Usage: dart run smollm2.dart '
          '[-m modelPath] '
          '[-n maxTokens] '
          '[-t temperature] '
          '[-r repeatPenalty] '
          '[-s seed] '
          '[-p prompt] '
          '[-nc no-colored] '
          '[-c chatMode]',
        );
        return;
    }
  }

  seed ??= SmolLM2.generateSeed();

  if (chatMode) {
    temperature ??= TokenGenerator.defaultChatTemperature;
    repeatPenalty ??= TokenGenerator.defaultChatRepeatPenalty;
  } else {
    temperature ??= TokenGenerator.defaultTemperature;
    repeatPenalty ??= TokenGenerator.defaultRepeatPenalty;
  }

  final smollm = SmolLM2(logger: (o) => print(' » $o'));

  print('=== SmolLM2 ===');
  print(
    '»» Parameters: maxTokens: $maxTokens ; temperature: $temperature ; repetitionPenalty: $repeatPenalty ; seed: $seed ; colored: $colored',
  );

  await smollm.load(model);

  if (chatMode) {
    await _chatSession(
      smollm,
      maxTokens,
      temperature,
      repeatPenalty,
      seed,
      colored,
    );
  } else {
    await _promptComplete(
      smollm,
      singlePrompt,
      maxTokens,
      temperature,
      repeatPenalty,
      seed,
      colored,
    );
  }
}

const _reset = '\x1B[0m';
const _gray = '\x1B[90m';
const _blue = '\x1B[34m';
const _cyan = '\x1B[36m';
const _yellow = '\x1B[33m';
const _red = '\x1B[31m';

Future<void> _promptComplete(
  SmolLM2 smollm,
  String? singlePrompt,
  int maxTokens,
  double temperature,
  double repeatPenalty,
  int seed,
  bool colored,
) async {
  print(
    _c(
      '---------------------------------------------------------',
      _gray,
      colored,
    ),
  );

  final prompt = singlePrompt ?? 'The capital of France is';

  void onTokenEmitted(int t, String s, TokenOrigin o) {
    stdout.write(s);
  }

  void onTokenEmittedColored(int t, String s, TokenOrigin o) {
    const reset = '\x1B[0m';

    final color = switch (o) {
      TokenOrigin.prompt => _cyan,
      TokenOrigin.generated => _blue,
      TokenOrigin.eos => _red,
      TokenOrigin.maxTokensReached => _yellow,
    };

    if (s.isEmpty && o.isTerminal) {
      s = '¤';
    }

    stdout.write('$color$s$reset');
  }

  var result = await smollm.generate(
    prompt,
    maxTokens: maxTokens,
    temperature: temperature,
    repeatPenalty: repeatPenalty,
    seed: seed,
    onTokenEmitted: colored ? onTokenEmittedColored : onTokenEmitted,
  );

  print('\n---------------------------------------------------------');

  print(result.statsSummary());
}

String _c(String text, String color, bool enabled) {
  if (!enabled) return text;
  return '$color$text$_reset';
}

Future<void> _chatSession(
  SmolLM2 smollm,
  int maxTokens,
  double temperature,
  double repeatPenalty,
  int seed,
  bool colored,
) async {
  final systemColor = _gray;
  final labelColor = _gray;
  final aiTextColor = _blue;

  void hr() {
    print(
      _c(
        '---------------------------------------------------------',
        systemColor,
        colored,
      ),
    );
  }

  final chat = ChatSession();

  chat.addSystem('You are a helpful AI assistant.');

  hr();
  print(_c('Chat mode enabled. Type "exit" to quit.', systemColor, colored));
  hr();

  var messagesOffset = 0;

  while (true) {
    stdout.write(_c('\nYou › ', labelColor, colored));
    final input = stdin.readLineSync();
    if (input == null) continue;

    if (input.trim().toLowerCase() == 'exit') {
      break;
    }

    chat.addUser(input);

    final assistantResponseBuffer = StringBuffer();

    void onTokenEmitted(int t, String s, TokenOrigin o) {
      stdout.write(_c(s, aiTextColor, colored));
      assistantResponseBuffer.write(s);
    }

    var nextPrompt = chat.buildPrompt(offset: messagesOffset);

    stdout.write(_c('\n AI › ', labelColor, colored));

    await smollm.generate(
      nextPrompt,
      maxTokens: maxTokens,
      temperature: temperature,
      repeatPenalty: repeatPenalty,
      seed: seed,
      emmitPromptTokens: false,
      onTokenEmitted: onTokenEmitted,
    );

    var assistantResponse = assistantResponseBuffer.toString();

    if (!assistantResponse.trim().endsWith('<|im_end|>')) {
      await smollm.ingest('<|im_end|>\n');
    }

    chat.addAssistant(assistantResponse);

    stdout.write('\n');

    messagesOffset = chat.length;
  }

  hr();
  print(_c('Full processed text:', systemColor, colored));
  hr();
  print('\n${smollm.fullText}');
  hr();
}
