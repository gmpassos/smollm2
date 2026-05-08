import 'dart:convert';

import 'package:smollm2/smollm2.dart';
import 'package:test/test.dart';

import 'smollm2_vocab.dart';

void main() {
  group('Tokenizer', () {
    test('builds vocab map correctly', () {
      final t = Tokenizer(vocab: ['a', 'b', 'ab'], merges: [], addedTokens: {});

      expect(t.vocabMap['a'], 0);
      expect(t.vocabMap['b'], 1);
      expect(t.vocabMap['ab'], 2);
    });

    test('findTok works', () {
      final t = Tokenizer(vocab: ['a', 'b'], merges: [], addedTokens: {});

      expect(t.findTok('a'), 0);
      expect(t.findTok('b'), 1);
      expect(t.findTok('x'), -1);
    });

    test('encodes space and newline correctly', () {
      final t = Tokenizer(
        vocab: ['\u0120', '\u010A'],
        merges: [],
        addedTokens: {},
      );

      final engine = TokenizerEngine(t);

      final result = engine.tokenize(' \n', 10);

      expect(result.length, 2);
      expect(result, contains(0)); // space
      expect(result, contains(1)); // newline
    });

    test('simple byte-level tokenization', () {
      final t = Tokenizer(vocab: ['a', 'b'], merges: [], addedTokens: {});

      final engine = TokenizerEngine(t);

      final result = engine.tokenize('ab', 10);

      expect(result, [0, 1]);
    });

    test('respects max length', () {
      final t = Tokenizer(vocab: ['a', 'b', 'c'], merges: [], addedTokens: {});

      final engine = TokenizerEngine(t);

      final result = engine.tokenize('abc', 2);

      expect(result.length, 2);
    });

    test('applies BPE merge when available', () {
      final t = Tokenizer(
        vocab: ['a', 'b', 'ab'],
        merges: [('a', 'b')],
        addedTokens: {},
      );

      final engine = TokenizerEngine(t);

      final result = engine.tokenize('ab', 10);

      // after merge should become single token "ab"
      expect(result.length, 1);
      expect(result[0], 2);
    });

    test('no merge when not in vocab', () {
      final t = Tokenizer(
        vocab: ['a', 'b'],
        merges: [('a', 'b')],
        addedTokens: {},
      );

      final engine = TokenizerEngine(t);

      final result = engine.tokenize('ab', 10);

      // cannot merge because "ab" not in vocab
      expect(result.length, 2);
    });

    test('utf8 multi-byte handling', () {
      final t = Tokenizer(vocab: ['é'], merges: [], addedTokens: {});

      final engine = TokenizerEngine(t);

      final result = engine.tokenize('é', 10);

      expect(result.length, greaterThanOrEqualTo(1));
    });

    test('encodes chat template correctly', () {
      final t = Tokenizer(
        vocab: smolLM2TokenizerVocabulary,
        merges: smolLM2TokenizerMerges,
        addedTokens: smolLM2TokenizerAddedTokens,
      );

      final engine = TokenizerEngine(t);

      final input = '''<|im_start|>system
You are a helpful AI assistant<|im_end|>
<|im_start|>user
My name is Alice.<|im_end|>
<|im_start|>assistant
Hello Alice.<|im_end|>
''';

      final result = engine.tokenize(input, 100);

      expect(result.isNotEmpty, true);

      final decoded = result.map((id) => engine.decode(id)).toList();
      print(json.encode(decoded));

      // Check structure markers exist
      expect(decoded.contains('<|im_start|>'), true);
      expect(decoded.contains('<|im_end|>'), true);

      expect(
        decoded,
        equals(
          json.decode(r'''
          [
          "<|im_start|>","s","y","s","t","e","m","\n",
          "Y","o","u"," ","a","r","e"," ","a"," ","h","e","l","p","f","u","l"," ","A","I"," ","a","s","s","i","s","t","a","n","t","<|im_end|>","\n",
          "<|im_start|>","u","s","e","r","\n","M","y"," ","n","a","m","e"," ","i","s"," ","A","l","i","c","e",".","<|im_end|>","\n",
          "<|im_start|>","a","s","s","i","s","t","a","n","t","\n","H","e","l","l","o"," ","A","l","i","c","e",".","<|im_end|>","\n"
          ]
          '''),
        ),
      );

      final decodedStr = decoded.join();
      print(decodedStr);

      expect(decodedStr, equals(input));
    });
  });
}
