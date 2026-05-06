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

  /// Builds a formatted prompt string from the stored messages.
  ///
  /// The output follows a chat template using special tokens:
  /// `<|im_start|>` and `<|im_end|>`, and appends a final assistant
  /// prompt to signal where the model should continue generation.
  ///
  /// The [offset] parameter allows skipping earlier messages,
  /// which is useful for sliding window context management.
  String buildPrompt({int offset = 0}) {
    final sb = StringBuffer();

    for (var i = offset; i < _messages.length; ++i) {
      final m = _messages[i];
      sb.write('<|im_start|>${m.role.name}\n');
      sb.write(m.content);
      sb.write('<|im_end|>\n');
    }

    sb.write('<|im_start|>assistant\n');

    return sb.toString();
  }
}
