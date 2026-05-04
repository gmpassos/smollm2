class Tokenizer {
  final int vocabSize;
  final int numMerges;

  final List<String> vocab;
  final List<String> merges;

  final Map<String, int> vocabMap = {};
  final List<MergePair> mergePairs = [];

  Tokenizer({required this.vocab, required this.merges})
    : vocabSize = vocab.length,
      numMerges = merges.length {
    _buildVocabMap();
    _buildMergePairs();
  }

  void _buildVocabMap() {
    final length = vocab.length;
    for (int i = 0; i < length; i++) {
      vocabMap[vocab[i]] = i;
    }
  }

  void _buildMergePairs() {
    for (final merge in merges) {
      final sp = merge.indexOf(' ');
      if (sp < 0) {
        continue;
      }

      var a = merge.substring(0, sp);
      var b = merge.substring(sp + 1);

      mergePairs.add(MergePair(a, b));
    }
  }

  int findTok(String s) => vocabMap[s] ?? -1;

  @override
  String toString() =>
      'Tokenizer{vocabSize: $vocabSize, numMerges: $numMerges}';
}

class MergePair {
  final String a;
  final String b;

  MergePair(this.a, this.b);
}
