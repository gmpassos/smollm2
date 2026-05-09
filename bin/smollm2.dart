import 'dart:io';

import 'package:smollm2/smollm2.dart';

Future<void> main(List<String> args) async {
  String model = 'models/smollm2-135m-instruct/smollm2-q16.bin';

  int maxTokens = 200;
  double? temperature;
  double? repeatPenalty;

  int? seed;

  bool jitterEnabled = false;
  int? jitterSeed;
  double? jitterScale;

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

      case '-j':
        jitterEnabled = true;
        break;
      case '-js':
        jitterEnabled = true;
        jitterSeed = int.parse(args[++i]);
        break;
      case '-jc':
        jitterEnabled = true;
        jitterScale = double.parse(args[++i]);
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
          '[-j] '
          '[-js jitterSeed] '
          '[-jc jitterScale] '
          '[-p prompt] '
          '[-nc no-colored] '
          '[-c chatMode]',
        );
        return;
    }
  }

  seed ??= SmolLM2.generateSeed();

  if (jitterEnabled) {
    jitterSeed ??= SmolLM2.generateSeed();
  }

  if (chatMode) {
    temperature ??= TokenGenerator.defaultChatTemperature;
    repeatPenalty ??= TokenGenerator.defaultChatRepeatPenalty;
  } else {
    temperature ??= TokenGenerator.defaultTemperature;
    repeatPenalty ??= TokenGenerator.defaultRepeatPenalty;
  }

  print('=== SmolLM2 ===');
  print(
    '»» Parameters: '
    'maxTokens: $maxTokens ; '
    'temperature: $temperature ; '
    'repetitionPenalty: $repeatPenalty ; '
    'seed: $seed ; '
    'jitter: ${jitterEnabled ? 'on' : 'off'} ; '
    'jitterSeed: ${jitterSeed ?? '-'} ; '
    'jitterScale: ${jitterScale ?? '-'} ; '
    'colored: $colored',
  );

  final smollm = SmolLM2(logger: (o) => print(' » $o'));

  await smollm.load(model, jitterSeed: jitterSeed, jitterScale: jitterScale);

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

enum SpinnerStyle {
  ascii(['|', '/', '-', '\\']),
  braille(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']),
  dots(['⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽', '⣾']),
  circle(['◐', '◓', '◑', '◒']),
  bars(['▁', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃']),
  blocks(['▖', '▘', '▝', '▗']),
  arrows(['←', '↖', '↑', '↗', '→', '↘', '↓', '↙']),
  dotsWave(['⠋', '⠙', '⠚', '⠞', '⠖', '⠦', '⠴', '⠲', '⠳', '⠓']);

  final List<String> frames;

  const SpinnerStyle(this.frames);
}

class TokenSpinner {
  final SpinnerStyle style;
  final String message;

  TokenSpinner(this.message, {this.style = SpinnerStyle.dots});

  int _i = 0;

  bool _active = false;

  void start() {
    _active = true;
    stdout.write(message);
  }

  void tick() {
    if (!_active) return;

    final frame = style.frames[_i++ % style.frames.length];
    stdout.write('\r$message $frame');
  }

  void stop([String? finalMessage]) {
    _active = false;
    stdout.write('\r\x1B[K${finalMessage ?? message}\n');
  }
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

  final chat = ChatSession(seed: seed);

  chat.addSystem(
    'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
  );

  hr();
  print(_c('Chat mode enabled. Type "exit" to quit.', systemColor, colored));
  hr();

  var messagesOffset = 0;

  var systemPrompt = chat.buildPrompt(
    offset: messagesOffset,
    appendImStartAssistant: false,
  );

  print('');

  TokenSpinner? spinner;
  if (colored) {
    spinner = TokenSpinner(
      _c('» Loading system prompt...', systemColor, colored),
    );
    spinner.start();
  } else {
    print('» Loading system prompt...');
  }

  smollm.ingest(
    systemPrompt,
    onTokenEmitted: (int t, String s, TokenOrigin o) {
      spinner?.tick();
    },
  );

  if (colored) {
    spinner!.stop(_c('» System prompt loaded', systemColor, colored));
  }

  ++messagesOffset;

  while (true) {
    stdout.write(_c('\nYou › ', labelColor, colored));
    final input = stdin.readLineSync();
    if (input == null) continue;

    if (input.trim().toLowerCase() == 'exit') {
      break;
    }

    chat.addUser(input);

    void onTokenEmitted(int t, String s, TokenOrigin o) {
      stdout.write(_c(s, aiTextColor, colored));
    }

    var nextPrompt = chat.buildPrompt(offset: messagesOffset);

    stdout.write(_c('\n AI › ', labelColor, colored));

    var result = await smollm.generate(
      nextPrompt,
      maxTokens: maxTokens,
      temperature: temperature,
      repeatPenalty: repeatPenalty,
      seed: seed,
      random: chat.random,
      includePromptInOutput: false,
      emmitPromptTokens: false,
      onTokenEmitted: onTokenEmitted,
    );

    var generatedTokens = result.generatedTokens;
    if (generatedTokens.isNotEmpty) {
      switch (result.stopReason) {
        case TokenGenerationStopReason.eosToken:
          {
            assert(smollm.tokenizer.isEOSTok(generatedTokens.last));
            smollm.ingest('\n', emmitPromptTokens: false);
          }
        case TokenGenerationStopReason.maxTokensReached:
          {
            assert(!smollm.tokenizer.isEOSTok(generatedTokens.last));
            smollm.ingest('${chat.imEnd}\n', emmitPromptTokens: false);
          }
      }
    }

    var assistantResponse = result.output;

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
