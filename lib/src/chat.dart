import 'dart:math' as math;

/// Defines the role of a message inside a chat conversation.
///
/// Used to distinguish between system instructions, user inputs,
/// and assistant responses when building prompts for LLMs.
enum ChatRole { system, user, assistant }

/// Represents a single message in a chat session.
///
/// Each message has a [role], which indicates who produced the message,
/// and a [content], which contains the actual text of the message.
class ChatMessage {
  /// The role of the message sender (system, user, or assistant).
  final ChatRole role;

  /// The textual content of the message.
  final String content;

  /// Creates a chat message with a given [role] and [content].
  ChatMessage(this.role, this.content);
}

/// Maintains a sequence of chat messages and can convert them into
/// a prompt format compatible with chat-based language models.
class ChatSession {
  static final _randomSecure = math.Random.secure();

  /// Generates a random seed suitable for deterministic or reproducible runs.
  ///
  /// The value is produced using a secure random generator and is guaranteed
  /// to be a positive 31-bit integer.
  ///
  /// Useful when the user does not explicitly provide a seed.
  static int generateSeed() {
    return _randomSecure.nextInt(0x7fffffff);
  }

  /// Random seed used for deterministic generation.
  ///
  /// When a seed is provided, the model will produce reproducible outputs
  /// for the same input and parameters.
  ///
  /// If not provided, a random seed is generated automatically.
  final int seed;

  /// Random number generator used for sampling and stochastic operations.
  ///
  /// This is initialized during construction and used across the session
  /// to ensure controlled randomness based on the configured seed.
  late final math.Random random;

  /// Token used to mark the beginning of a message block in the chat template.
  ///
  /// Typically used as part of the model's chat formatting protocol:
  /// `<|im_start|>role\ncontent<|im_end|>`
  static const defaultIMStart = '<|im_start|>';

  /// Token used to mark the end of a message block in the chat template.
  ///
  /// Paired with [defaultIMStart] to delimit structured chat messages.
  static const defaultIMEnd = '<|im_end|>';

  /// Start token used to delimit chat message blocks.
  ///
  /// Overrides [defaultIMStart] if provided.
  final String imStart;

  /// End token used to delimit chat message blocks.
  ///
  /// Overrides [defaultIMEnd] if provided.
  final String imEnd;

  /// Creates a new [ChatSession].
  ///
  /// If [seed] is not provided, a random seed is generated automatically.
  /// The seed controls randomness and ensures reproducible generation when fixed.
  ///
  /// [imStart] and [imEnd] define the chat template delimiters used to format
  /// system, user, and assistant messages.
  ChatSession({
    int? seed,
    this.imStart = ChatSession.defaultIMStart,
    this.imEnd = ChatSession.defaultIMEnd,
  }) : seed = seed ?? generateSeed() {
    random = math.Random(this.seed);
  }

  /// Internal list of messages in the session, in chronological order.
  final List<ChatMessage> _messages = [];

  /// Number of messages currently stored in the session.
  int get length => _messages.length;

  /// Adds a system-level instruction message.
  void addSystem(String text) =>
      _messages.add(ChatMessage(ChatRole.system, text));

  /// Adds a user message to the session.
  void addUser(String text) => _messages.add(ChatMessage(ChatRole.user, text));

  /// Adds an assistant message to the session.
  void addAssistant(String text) =>
      _messages.add(ChatMessage(ChatRole.assistant, text));

  /// Builds a chat-formatted prompt from the stored message history.
  ///
  /// The output follows a structured template compatible with
  /// ChatML-style models, using [imStart] and [imEnd] tokens
  /// to delimit each message block.
  ///
  /// Each message is formatted as:
  ///
  /// ```text
  /// <|im_start|>role
  /// content
  /// <|im_end|>
  /// ```
  ///
  /// After all messages, an optional assistant prefix is appended:
  ///
  /// ```text
  /// <|im_start|>assistant
  /// ```
  ///
  /// This signals the model to continue generation from the assistant role.
  ///
  /// The [offset] parameter allows skipping older messages, enabling
  /// sliding-window or truncated-context strategies for long conversations.
  String buildPrompt({int offset = 0, bool appendImStartAssistant = true}) {
    final sb = StringBuffer();

    for (var i = offset; i < _messages.length; ++i) {
      final m = _messages[i];
      sb.write('$imStart${m.role.name}\n');
      sb.write(m.content);
      sb.write('$imEnd\n');
    }

    if (appendImStartAssistant) {
      sb.write('${imStart}assistant\n');
    }

    return sb.toString();
  }

  /// Checks whether the given [text] ends with the configured [imEnd] token.
  ///
  /// This is used to determine if a generated assistant response has already been
  /// properly terminated according to the chat template format.
  ///
  /// The check ignores trailing whitespace before evaluating the suffix.
  bool endsWithImEndToken(String text) {
    return text.trim().endsWith(imEnd);
  }
}
