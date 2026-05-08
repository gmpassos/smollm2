# SmolLM2 Examples

The `example/` directory contains practical examples showing how to use the `smollm2` package for text completion and
chat-based inference with local SmolLM2 models.

GitHub repository:

[smollm2 GitHub Repository](https://github.com/gmpassos/smollm2)

Examples directory:

[example/ Directory](https://github.com/gmpassos/smollm2/tree/master/example)

---

# Available Examples

| File                                    | Description                                                                 |
|-----------------------------------------|-----------------------------------------------------------------------------|
| `smollm2_completion_example.dart`       | Basic text completion example                                               |
| `smollm2_chat_example.dart`             | Interactive multi-turn chat session                                         |
| `smollm2_rs_in_strawberry_example.dart` | Prompt formatting and step-by-step reasoning example (“r”s in “strawberry”) |

---

# Models

These examples expect exported SmolLM2 models in the `models/` directory.

Example model path:

```text
models/smollm2-360m-instruct/smollm2-bf16.bin
```

---

# Running an Example

Run any example using:

```bash
dart run example/smollm2_completion_example.dart
```

Or specify a custom model path:

```bash
dart run example/smollm2_completion_example.dart \
  models/smollm2-360m-instruct/smollm2-bf16.bin
```

---

# Text Completion Example

File:

```text
example/smollm2_completion_example.dart
```

This example demonstrates:

* Loading a SmolLM2 model
* Generating text completions
* Sampling configuration
* Deterministic generation using a seed

```dart
import 'package:smollm2/smollm2.dart';

Future<void> main(List<String> args) async {
  // In this example we use the 360m Instruct BF16 model.
  var modelPath = args.isNotEmpty
      ? args[0]
      : 'models/smollm2-360m-instruct/smollm2-bf16.bin';

  // Create a new SmolLM2 inference engine instance.
  final smollm = SmolLM2(logger: (o) => print('»» $o'));

  // Load the exported SmolLM2 model into memory.
  await smollm.load(modelPath);

  // Prompt to start the text generation.
  var prompt = 'The capital of France is';

  print('---------------------------------------------------');

  // Generate text directly to stdout using the configured sampling options:
  // - maxTokens: maximum number of tokens to generate
  // - temperature: controls randomness (lower = more deterministic)
  // - repeatPenalty: discourages repetitive output
  // - seed: ensures deterministic generation for reproducible results
  var result = await smollm.generate(
    prompt,
    maxTokens: 60,
    temperature: 0.2,
    repeatPenalty: 1.1,
    seed: 123456, // not used for `temperature: 0.0`
  );

  // The token generation result.output:
  print('\n<<<\n${result.output}\n>>>');

  // The actual steam of text processed by the LLM:
  print('\n<<<\n${smollm.fullText}\n>>>');
}
```

Example output:

```text
The capital of France is Paris.
Paris is the largest city in France.
Paris has a rich history and culture, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
```

---

# Interactive Chat Example

File:

```text
example/smollm2_chat_example.dart
```

This example demonstrates:

* Stateful chat conversations
* Prompt construction using `ChatSession`
* Streaming token output
* Multi-turn conversations
* Persistent chat history

```dart
import 'dart:io';

import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  final smollm = SmolLM2();
  await smollm.load('models/smollm2-360m-instruct/smollm2-bf16.bin');

  final chat = ChatSession(seed: 12345);
  chat.addSystem('You are a helpful assistant.');

  var messagesOffset = 0;

  void onTokenEmitted(int t, String s, TokenOrigin o) {
    stdout.write(s);
  }

  print('[Chat ready. Type "exit" to quit]');

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
```

Example session:

```text
[Chat ready. Type "exit" to quit]

You › Hello!
 AI › Hello! How can I help you today?

You › What is Dart?
 AI › Dart is a general-purpose, statically-typed, multi-paradigm language developed by Google.
```

---

# Prompt Formatting Example

File:

```text
example/smollm2_rs_in_strawberry_example.dart
```

This example demonstrates:

* Manual prompt formatting
* SmolLM2 instruction template usage
* System/user/assistant role formatting
* Simple reasoning prompts

```dart
import 'package:smollm2/smollm2.dart';

Future<void> main() async {
  // Create a new SmolLM2 inference engine instance.
  final smollm = SmolLM2(logger: (o) => print('»» $o'));

  // Load the exported SmolLM2 model into memory.
  // In this example we use the 360m Instruct BF16 model.
  await smollm.load('models/smollm2-360m-instruct/smollm2-bf16.bin');

  var prompt = '''<|im_start|>system
You are a helpful AI assistant<|im_end|>
<|im_start|>user
How many r's in Strawberry?<|im_end|>
<|im_start|>assistant
''';

  print('---------------------------------------------------');

  var output = await smollm.generate(
    prompt,
    maxTokens: 40,
    temperature: 0.1,
    repeatPenalty: 1.0,
    seed: 12345,
  );

  print('\n<<<\n$output\n>>>');
}
```

Example output:

```text
There are 3 r's in the word "Strawberry."
```

---

# Notes

* Lower `temperature` values produce more deterministic outputs.
* `repeatPenalty` helps reduce repetitive generation.
* `seed` allows reproducible inference results.
* `ChatSession` simplifies prompt management for multi-turn conversations.
* `smollm.fullText` contains the complete processed token stream.

---

# See Also

* [smollm2 Package Repository](https://github.com/gmpassos/smollm2)
* [example/ Source Files](https://github.com/gmpassos/smollm2/tree/master/example)
