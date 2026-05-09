import 'dart:io';

import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  final smollm = SmolLM2();
  await smollm.load('models/smollm2-360m-instruct/smollm2-bf16.bin');

  final chat = ChatSession(seed: 12345);

  var messagesOffset = 0;

  chat.addSystem('You are a helpful assistant.');

  print('Loading system prompt...');
  var systemPrompt = chat.buildPrompt(
    offset: messagesOffset,
    appendImStartAssistant: false,
  );

  smollm.ingest(systemPrompt);

  ++messagesOffset;

  void onTokenEmitted(int t, String s, TokenOrigin o) {
    stdout.write(s);
  }

  print('\n[Chat ready. Type "exit" to quit]');

  while (true) {
    stdout.write('\nYou › ');
    final input = stdin.readLineSync();
    if (input == null) continue;

    if (input.trim().toLowerCase() == 'exit') break;

    chat.addUser(input);

    final prompt = chat.buildPrompt(offset: messagesOffset);

    stdout.write(' AI › ');

    var result = await smollm.generate(
      prompt,
      includePromptInOutput: false,
      emmitPromptTokens: false,
      temperature: TokenGenerator.defaultChatTemperature,
      repeatPenalty: TokenGenerator.defaultChatRepeatPenalty,
      random: chat.random,
      onTokenEmitted: onTokenEmitted,
    );

    final assistantText = result.output;
    chat.addAssistant(assistantText);

    switch (result.stopReason) {
      case TokenGenerationStopReason.eosToken:
        {
          smollm.ingest('\n', emmitPromptTokens: false);
        }
      case TokenGenerationStopReason.maxTokensReached:
        {
          smollm.ingest('${chat.imEnd}\n', emmitPromptTokens: false);
        }
    }

    messagesOffset = chat.length;

    stdout.write('\n');
  }

  print('----------------------------------------------------');
  print('Full processed text:\n');
  print(smollm.fullText);
}

// OUTPUT:
/*
Loading system prompt...

[Chat ready. Type "exit" to quit]

You › Hello!
 AI › Hello! How can I help you today?

You › What is Dart?
 AI › Dart is a general-purpose, statically-typed, multi-paradigm language developed by Google. It's used for web development, mobile app development, and Android app development.

You › exit
----------------------------------------------------
Full processed text:

<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hello! How can I help you today?<|im_end|>
<|im_start|>user
What is Dart?<|im_end|>
<|im_start|>assistant
Dart is a general-purpose, statically-typed, multi-paradigm language developed by Google. It's used for web development, mobile app development, and Android app development.<|im_end|>

*/
