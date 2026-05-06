import 'dart:io';

import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  final smollm = SmolLM2();
  await smollm.load('models/smollm2-135m-instruct/smollm2-q8.bin');

  final chat = ChatSession(seed: 12345);
  chat.addSystem('You are a helpful assistant.');

  var messagesOffset = 0;

  void onTokenEmitted(int t, String s, TokenOrigin o) {
    stdout.write(s);
  }

  print('Chat ready. Type "exit" to quit.');

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

    if (!chat.endsWithImEndToken(assistantText)) {
      await smollm.ingest('${chat.imEnd}\n');
    }

    messagesOffset = chat.length;

    stdout.write('\n');
  }

  print('----------------------------------------------------');
  print('Full processed text:\n');
  print(smollm.fullText);
}
