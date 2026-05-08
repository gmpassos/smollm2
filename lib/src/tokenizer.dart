import 'data.dart';

class Tokenizer {
  final int vocabSize;
  final int numMerges;
  final int addedSize;

  final List<String> vocab;
  final List<(String, String)> merges;
  final Map<String, int> addedTokens;

  final Map<String, int> vocabMap = {};
  final Map<(String, String), int> mergeRank = {};

  Tokenizer({
    required this.vocab,
    required this.merges,
    required this.addedTokens,
  }) : vocabSize = vocab.length,
       numMerges = merges.length,
       addedSize = addedTokens.length {
    _buildVocabMap();
    _buildMergeRank();
  }

  factory Tokenizer.loadFrom(DataReader dataReader) {
    final vocabLength = dataReader.readU32();
    final mergesLength = dataReader.readU32();

    final vocab = List.generate(vocabLength, (_) => dataReader.readString());

    final merges = List.generate(mergesLength, (_) {
      final a = dataReader.readString();
      final b = dataReader.readString();
      return (a, b);
    });

    // --- NEW: added tokens ---
    final addedCount = dataReader.readU32();
    final addedTokens = <String, int>{};

    for (int i = 0; i < addedCount; i++) {
      final text = dataReader.readString();
      final id = dataReader.readU32();
      addedTokens[text] = id;
    }

    final t = Tokenizer(vocab: vocab, merges: merges, addedTokens: addedTokens);

    return t;
  }

  void _buildVocabMap() {
    for (int i = 0; i < vocab.length; i++) {
      vocabMap[vocab[i]] = i;
    }
  }

  void _buildMergeRank() {
    for (int i = 0; i < merges.length; i++) {
      mergeRank[merges[i]] = i; // lower = higher priority
    }
  }

  int findTok(String s) => vocabMap[s] ?? -1;

  @override
  String toString() =>
      'Tokenizer{vocabSize: $vocabSize, numMerges: $numMerges}';
}

class TokenizerEngine {
  final Tokenizer tokenizer;

  TokenizerEngine(this.tokenizer);

  List<int> tokenize(String text, int max) {
    final t = tokenizer;

    final toks = <int>[];

    final added = t.addedTokens; // Map<String, int>

    int i = 0;
    while (i < text.length) {
      int? matchedId;
      int matchedLen = 0;

      // 1. MATCH ADDED TOKENS (longest match wins)
      for (final entry in added.entries) {
        final sp = entry.key;
        if (sp.isEmpty) continue;

        if (i + sp.length <= text.length && text.startsWith(sp, i)) {
          if (sp.length > matchedLen) {
            matchedId = entry.value;
            matchedLen = sp.length;
          }
        }
      }

      if (matchedId != null) {
        toks.add(matchedId);
        i += matchedLen;
        if (toks.length >= max) return toks;
        continue;
      }

      // 2. NORMAL CHARACTER HANDLING
      final ch = text[i];

      if (ch == ' ') {
        final id = t.findTok('\u0120');
        if (id >= 0) toks.add(id);
      } else if (ch == '\n') {
        final id = t.findTok('\u010A');
        if (id >= 0) toks.add(id);
      } else {
        final id = t.findTok(ch);
        if (id >= 0) toks.add(id);
      }

      i++;
      if (toks.length >= max) return toks;
    }

    // 3. BPE MERGE (unchanged)
    while (toks.length > 1) {
      int bestIndex = -1;
      int bestRank = 1 << 30;

      for (int i = 0; i < toks.length - 1; i++) {
        final a = t.vocab[toks[i]];
        final b = t.vocab[toks[i + 1]];

        final rank = t.mergeRank[(a, b)];
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }

      if (bestIndex == -1) break;

      final merged = t.vocab[toks[bestIndex]] + t.vocab[toks[bestIndex + 1]];

      final mergedId = t.findTok(merged);
      if (mergedId < 0) break;

      toks[bestIndex] = mergedId;
      toks.removeAt(bestIndex + 1);

      if (toks.length >= max) break;
    }

    return toks;
  }

  String decode(int tok) {
    final tokenizer = this.tokenizer;

    if (tok < 0 || tok >= tokenizer.vocabSize) {
      return '';
    }

    final raw = tokenizer.vocab[tok];
    final out = StringBuffer();

    for (int i = 0; i < raw.length; i++) {
      final ch = raw[i];

      if (ch == '\u0120') {
        out.write(' ');
      } else if (ch == '\u010A') {
        out.write('\n');
      } else {
        out.write(ch);
      }
    }

    return out.toString();
  }
}
