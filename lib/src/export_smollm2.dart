import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:collection/collection.dart';

import 'data.dart';
import 'quant_type.dart';

/* ---------------- Config ---------------- */

class HFConfig {
  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int vocabSize;
  final int maxPositionEmbeddings;
  final double ropeTheta;
  final double rmsNormEps;
  final int bosTokenId;
  final int eosTokenId;

  HFConfig({
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.vocabSize,
    required this.maxPositionEmbeddings,
    required this.ropeTheta,
    required this.rmsNormEps,
    required this.bosTokenId,
    required this.eosTokenId,
  });

  factory HFConfig.fromJson(Map<String, dynamic> json) {
    return HFConfig(
      hiddenSize: json['hidden_size'],
      intermediateSize: json['intermediate_size'],
      numHiddenLayers: json['num_hidden_layers'],
      numAttentionHeads: json['num_attention_heads'],
      numKeyValueHeads: json['num_key_value_heads'],
      vocabSize: json['vocab_size'],
      maxPositionEmbeddings: json['max_position_embeddings'],
      ropeTheta: (json['rope_theta'] as num).toDouble(),
      rmsNormEps: (json['rms_norm_eps'] as num).toDouble(),
      bosTokenId: json['bos_token_id'] ?? 1,
      eosTokenId: json['eos_token_id'] ?? 2,
    );
  }
}

/* ---------------- Tokenizer ---------------- */

class HFTokenizer {
  final List<String> vocab;
  final List<(String, String)> merges;

  HFTokenizer(this.vocab, this.merges);

  static Future<HFTokenizer> load(String path) async {
    final jsonMap = jsonDecode(await File(path).readAsString());
    final model = jsonMap['model'] as Map<String, dynamic>;
    final vocabMap = model['vocab'] as Map<String, dynamic>;
    final mergesGeneric = (model['merges'] as List);

    final List<(String, String)> merges = mergesGeneric.map((e) {
      if (e is List) {
        if (e.length == 2) {
          return (e[0].toString(), e[1].toString());
        } else {
          throw StateError(
            "Can't parse Tokenizer `merges` List entry of length: ${e.length}",
          );
        }
      } else if (e is String) {
        var idx = e.indexOf(' ');
        if (idx >= 0) {
          var a = e.substring(0, idx);
          var b = e.substring(idx + 1);
          return (a, b);
        } else {
          throw StateError(
            "Can't parse Tokenizer `merges` String entry: <<$e>>",
          );
        }
      } else {
        throw StateError(
          "Can't parse Tokenizer `merges` entry (${e.runtimeType}): <<$e>>",
        );
      }
    }).toList();

    final vocab = List<String>.filled(vocabMap.length, '');

    for (final e in vocabMap.entries) {
      vocab[e.value as int] = e.key;
    }

    return HFTokenizer(vocab, merges);
  }
}

/* ---------------- Tensor Metadata ---------------- */

class SafeTensorInfo {
  final String tensorName;
  final String dtype;
  final List<int> shape;
  final int begin;
  final int end;

  const SafeTensorInfo(
    this.tensorName,
    this.dtype,
    this.shape,
    this.begin,
    this.end,
  );

  int get count {
    var c = 1;
    for (final d in shape) {
      c *= d;
    }
    return c;
  }
}

/* ---------------- Quant structs ---------------- */

abstract class Quantized {
  List<int> get data;

  (int, int)? _dataHash;

  (int, int) get dataHash => _dataHash ??= computeDataHash();

  (int, int) computeDataHash() => data.hashListInt2();
}

class Q8Quantized extends Quantized {
  final double scale;
  @override
  final Int8List data;

  Q8Quantized(this.scale, this.data);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Q8Quantized &&
          runtimeType == other.runtimeType &&
          scale == other.scale &&
          ListEquality().equals(data, other.data);

  @override
  int get hashCode => Object.hash(scale, data.hashListInt());

  @override
  String toString() => 'Q8Quantized{scale: $scale, data: ${data.length}}';
}

abstract class QuantizedPerBlock<B extends Quantized> extends Quantized {
  final int blockSize;
  final List<B> blocks;

  QuantizedPerBlock(this.blockSize, this.blocks);

  @override
  List<int> get data => blocks.expand((b) => b.data).toList();

  @override
  (int, int) computeDataHash() {
    return blocks
        .map((e) => e.dataHash)
        .reduce((a, b) => (a.$1 ^ b.$1, a.$2 ^ b.$2));
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is QuantizedPerBlock<B> &&
          runtimeType == other.runtimeType &&
          blockSize == other.blockSize &&
          const ListEquality().equals(blocks, other.blocks);

  @override
  int get hashCode => Object.hash(blockSize, const ListEquality().hash(blocks));
}

class Q8QuantizedPerBlock extends QuantizedPerBlock<Q8Quantized> {
  Q8QuantizedPerBlock(super.blockSize, super.blocks);

  @override
  String toString() =>
      'Q8QuantizedPerBlock{blockSize: $blockSize, blocks: ${blocks.length}}';
}

class Q16Quantized extends Quantized {
  final double scale;
  @override
  final Int16List data;

  Q16Quantized(this.scale, this.data);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Q16Quantized &&
          runtimeType == other.runtimeType &&
          scale == other.scale &&
          data == other.data;

  @override
  int get hashCode => Object.hash(scale, data.hashListInt());

  @override
  String toString() => 'Q16Quantized{scale: $scale, data: ${data.length}}';
}

class Q16QuantizedPerBlock extends QuantizedPerBlock<Q16Quantized> {
  Q16QuantizedPerBlock(super.blockSize, super.blocks);

  @override
  String toString() =>
      'Q16QuantizedPerBlock{blockSize: $blockSize, blocks: ${blocks.length}}';
}

/* ---------------- Decoders ---------------- */

abstract class TensorDTypeDecoder {
  Float32List decodeTensor(Uint8List bytes);

  double decodeScalar(Uint8List raw, int bytesOffset, int index);
}

/* F32 */
class F32Decoder implements TensorDTypeDecoder {
  @override
  Float32List decodeTensor(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);
    final out = Float32List(bytes.length ~/ 4);

    for (int i = 0; i < out.length; i++) {
      out[i] = bd.getFloat32(i * 4, Endian.little);
    }
    return out;
  }

  @override
  double decodeScalar(Uint8List raw, int bytesOffset, int index) {
    final valueOffset = bytesOffset + (index * 4);
    return ByteData.sublistView(raw).getFloat32(valueOffset, Endian.little);
  }
}

/* F16 */
class F16Decoder implements TensorDTypeDecoder {
  @override
  Float32List decodeTensor(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);
    final out = Float32List(bytes.length ~/ 2);

    for (int i = 0; i < out.length; i++) {
      out[i] = _decodeHalf(bd.getUint16(i * 2, Endian.little));
    }
    return out;
  }

  @override
  double decodeScalar(Uint8List raw, int bytesOffset, int index) {
    final valueOffset = bytesOffset + (index * 2);
    final h = ByteData.sublistView(raw).getUint16(valueOffset, Endian.little);
    return _decodeHalf(h);
  }

  double _decodeHalf(int h) {
    final s = (h >> 15) & 1;
    final e = (h >> 10) & 0x1F;
    final f = h & 0x3FF;

    double v;

    if (e == 0) {
      v = math.pow(2, -14) * (f / 1024.0);
    } else if (e == 31) {
      if (f == 0) return double.infinity;
      return double.nan;
    } else {
      v = math.pow(2, e - 15) * (1.0 + f / 1024.0);
    }

    return s == 1 ? -v : v;
  }
}

/* BF16 */
class BF16Decoder implements TensorDTypeDecoder {
  @override
  Float32List decodeTensor(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);
    final out = Float32List(bytes.length ~/ 2);

    final buffer = ByteData(4);

    for (int i = 0; i < out.length; i++) {
      var valueOffset = i * 2;

      // copy raw BF16 bytes directly into high half of FP32
      buffer.setUint8(2, bd.getUint8(valueOffset));
      buffer.setUint8(3, bd.getUint8(valueOffset + 1));

      out[i] = buffer.getFloat32(0, Endian.little);
    }

    return out;
  }

  @override
  double decodeScalar(Uint8List raw, int bytesOffset, int index) {
    var bd = ByteData.sublistView(raw);
    final buffer = ByteData(4);

    final valueOffset = bytesOffset + (index * 2);

    // copy raw BF16 bytes directly into high half of FP32
    buffer.setUint8(2, bd.getUint8(valueOffset));
    buffer.setUint8(3, bd.getUint8(valueOffset + 1));

    return buffer.getFloat32(0, Endian.little);
  }
}

/* Q16 */
class Q16Decoder implements TensorDTypeDecoder {
  @override
  Float32List decodeTensor(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);

    final scale = bd.getFloat32(0, Endian.little);
    final outLen = (bytes.length - 4) ~/ 2;
    final out = Float32List(outLen);

    int offset = 4;

    for (int i = 0; i < outLen; i++) {
      final v = bd.getInt16(offset, Endian.little);
      out[i] = v * scale;
      offset += 2;
    }

    return out;
  }

  @override
  double decodeScalar(Uint8List raw, int bytesOffset, int index) {
    final bd = ByteData.sublistView(raw);

    final scale = bd.getFloat32(bytesOffset, Endian.little);

    final valueOffset = bytesOffset + 4 + (index * 2);
    final v = bd.getInt16(valueOffset, Endian.little);

    return v * scale;
  }
}

/* ---------------- Factory ---------------- */

class TensorDecoderFactory {
  static TensorDTypeDecoder create(String dtype) {
    switch (dtype) {
      case 'F32':
        return F32Decoder();
      case 'F16':
        return F16Decoder();
      case 'BF16':
        return BF16Decoder();
      case 'Q16':
        return Q16Decoder();
      default:
        throw UnsupportedError('Unsupported dtype $dtype');
    }
  }
}

/* ---------------- Repository ---------------- */

abstract class TensorRepository {
  FutureOr<SafeTensorInfo> info(String tensorName);

  FutureOr<Float32List> readTensor(String tensorName);

  FutureOr<double> readScalar(String tensorName, int index);
}

class SafeTensorFileRepository implements TensorRepository {
  final Uint8List raw;
  final Map<String, SafeTensorInfo> tensors;

  SafeTensorFileRepository(this.raw, this.tensors);

  static Future<SafeTensorFileRepository> load(String path) async {
    final raw = await File(path).readAsBytes();

    final bd = ByteData.sublistView(raw);
    final headerLen = bd.getUint64(0, Endian.little);

    final headerJson = utf8.decode(raw.sublist(8, 8 + headerLen));
    final map = jsonDecode(headerJson) as Map<String, dynamic>;

    final tensors = <String, SafeTensorInfo>{};

    for (final e in map.entries) {
      if (e.key == '__metadata__') continue;

      final obj = e.value as Map<String, dynamic>;
      final offsets = (obj['data_offsets'] as List).cast<int>();

      tensors[e.key] = SafeTensorInfo(
        e.key,
        obj['dtype'],
        (obj['shape'] as List).cast<int>(),
        8 + headerLen + offsets[0],
        8 + headerLen + offsets[1],
      );
    }

    return SafeTensorFileRepository(raw, tensors);
  }

  @override
  SafeTensorInfo info(String tensorName) =>
      tensors[tensorName] ?? (throw ArgumentError('No tensor: $tensorName'));

  @override
  Float32List readTensor(String tensorName) {
    final meta = info(tensorName);
    final decoder = TensorDecoderFactory.create(meta.dtype);
    return decoder.decodeTensor(raw.sublist(meta.begin, meta.end));
  }

  @override
  double readScalar(String tensorName, int index) {
    final meta = info(tensorName);
    final decoder = TensorDecoderFactory.create(meta.dtype);
    return decoder.decodeScalar(raw, meta.begin, index);
  }
}

class SafeTensorShardRepository implements TensorRepository {
  final String baseDir;
  final Map<String, String> tensorToFile;
  final Map<String, SafeTensorFileRepository> cache = {};

  SafeTensorShardRepository(this.baseDir, this.tensorToFile);

  static Future<SafeTensorShardRepository> load(String indexPath) async {
    final file = File(indexPath);
    final dir = file.parent.path;

    final map = jsonDecode(await file.readAsString());
    final wm = map['weight_map'] as Map<String, dynamic>;

    final tensorToFile = <String, String>{};

    for (final e in wm.entries) {
      tensorToFile[e.key] = e.value as String;
    }

    return SafeTensorShardRepository(dir, tensorToFile);
  }

  Future<SafeTensorFileRepository> _repoOf(String tensorName) async {
    final fileName = tensorToFile[tensorName];
    if (fileName == null) {
      throw Exception('Tensor not found in index: $tensorName');
    }

    final cached = cache[fileName];
    if (cached != null) return cached;

    final repo = await SafeTensorFileRepository.load('$baseDir/$fileName');
    cache[fileName] = repo;
    return repo;
  }

  @override
  Future<SafeTensorInfo> info(String tensorName) async {
    return (await _repoOf(tensorName)).info(tensorName);
  }

  @override
  Future<Float32List> readTensor(String tensorName) async {
    return (await _repoOf(tensorName)).readTensor(tensorName);
  }

  @override
  Future<double> readScalar(String tensorName, int index) async {
    return (await _repoOf(tensorName)).readScalar(tensorName, index);
  }
}

/* ---------------- Quantizers ---------------- */
class Q8Quantizer {
  Q8Quantized quantize(Float32List src) {
    final length = src.length;
    var data = Int8List(length);
    final scale = _quantizeRange(src, 0, length, data, 0);
    return Q8Quantized(scale, data);
  }

  double _quantizeRange(
    Float32List block,
    int offset,
    int end,
    Int8List out,
    int outOffset,
  ) {
    assert(end - offset > 0);

    double maxAbs = 0.0;
    for (int i = offset; i < end; i++) {
      final v = block[i];
      final a = v.abs();
      if (a > maxAbs) maxAbs = a;
    }

    final scale = maxAbs == 0 ? 1.0 : maxAbs / 127.0;

    for (int i = offset; i < end; i++) {
      final q = (block[i] / scale).round();
      out[outOffset++] = q.clamp(-127, 127);
    }

    return scale;
  }

  Q8QuantizedPerBlock quantizePerBlock(Float32List src, {int blockSize = 32}) {
    final blocks = <Q8Quantized>[];

    final length = src.length;
    var dataFull = Int8List(length);

    for (int offset = 0; offset < length; offset += blockSize) {
      final end = math.min(offset + blockSize, length);

      final blockScale = _quantizeRange(src, offset, end, dataFull, offset);

      final quantized = Q8Quantized(
        blockScale,
        Int8List.sublistView(dataFull, offset, end),
      );

      blocks.add(quantized);
    }

    return Q8QuantizedPerBlock(blockSize, blocks);
  }
}

class Q16Quantizer {
  Q16Quantized quantize(Float32List src) {
    final length = src.length;
    var data = Int16List(length);
    final scale = _quantizeRange(src, 0, length, data, 0);
    return Q16Quantized(scale, data);
  }

  double _quantizeRange(
    Float32List block,
    int offset,
    int end,
    Int16List out,
    int outOffset,
  ) {
    assert(end - offset > 0);

    double maxAbs = 0.0;
    for (int i = offset; i < end; i++) {
      final v = block[i];
      final a = v.abs();
      if (a > maxAbs) maxAbs = a;
    }

    final scale = maxAbs == 0 ? 1.0 : maxAbs / 32767.0;

    for (int i = offset; i < end; i++) {
      final q = (block[i] / scale).round();
      out[outOffset++] = q.clamp(-32767, 32767);
    }

    return scale;
  }

  Q16QuantizedPerBlock quantizePerBlock(Float32List src, {int blockSize = 32}) {
    final blocks = <Q16Quantized>[];

    final length = src.length;
    var dataFull = Int16List(length);

    for (int offset = 0; offset < length; offset += blockSize) {
      final end = math.min(offset + blockSize, length);
      var blockScale = _quantizeRange(src, offset, end, dataFull, offset);

      var quantized = Q16Quantized(
        blockScale,
        Int16List.sublistView(dataFull, offset, end),
      );

      blocks.add(quantized);
    }

    return Q16QuantizedPerBlock(blockSize, blocks);
  }
}

/* ---------------- Writer ---------------- */

class TensorBinaryWriter {
  final DataWriter dataWriter;

  final Q8Quantizer q8 = Q8Quantizer();
  final Q16Quantizer q16 = Q16Quantizer();

  TensorBinaryWriter(this.dataWriter);

  Future<void> writeTensor(Float32List tensor, QuantType type) async {
    switch (type) {
      case QuantType.q8:
        //await writeTensorQ8PerBlock(tensor);
        writeTensorQ8(tensor);
        break;

      case QuantType.q16:
        writeTensorQ16(tensor);
        break;

      case QuantType.q16PerBlock:
        writeTensorQ16PerBlock(tensor);
        break;

      case QuantType.bf16:
        writeTensorBF16H(tensor);
        break;

      case QuantType.fp32:
        writeTensorFP32(tensor);
        break;
    }

    await dataWriter.flush();
  }

  void writeTensorQ8(Float32List tensor) {
    final q = q8.quantize(tensor);
    final dataHash = q.dataHash;
    dataWriter.writeF32(q.scale);
    dataWriter.writeU32(dataHash.$1);
    dataWriter.writeU32(dataHash.$2);
    dataWriter.writeBytes(q.data.buffer.asUint8List());
  }

  Future<void> writeTensorQ8PerBlock(Float32List tensor) async {
    final q = q8.quantizePerBlock(tensor);

    await dataWriter.writeU16(q.blockSize);

    for (final block in q.blocks) {
      final data = block.data;
      final len = data.length;
      final hash = block.dataHash;

      // 4 bytes (f32) + 4 + 4 (u32 + u32) + data (len * 1) (int8)
      final bytes = Uint8List(12 + len);
      final bd = ByteData.sublistView(bytes);

      int offset = 0;

      // scale (f32)
      bd.setFloat32(offset, block.scale, Endian.little);
      offset += 4;

      // hash1
      bd.setUint32(offset, hash.$1, Endian.little);
      offset += 4;

      // hash2
      bd.setUint32(offset, hash.$2, Endian.little);
      offset += 4;

      // int8 data (bulk copy, no loop)
      bytes.setRange(offset, offset + len, data);

      dataWriter.writeBytes(bytes);
    }
  }

  void writeTensorQ16(Float32List tensor) {
    final q = q16.quantize(tensor);
    final dataHash = q.dataHash;

    // scale:
    dataWriter.writeF32(q.scale);

    // hash:
    dataWriter.writeU32(dataHash.$1);
    dataWriter.writeU32(dataHash.$2);

    // data
    final bd = ByteData(q.data.length * 2);
    for (int i = 0; i < q.data.length; i++) {
      bd.setInt16(i * 2, q.data[i], Endian.little);
    }

    dataWriter.writeBytes(bd.buffer.asUint8List());
  }

  void writeTensorQ16PerBlock(Float32List tensor) {
    final q = q16.quantizePerBlock(tensor);

    dataWriter.writeU16(q.blockSize);

    final blocks = q.blocks;
    for (var b = 0; b < blocks.length; ++b) {
      final block = blocks[b];

      final data = block.data;
      final len = data.length;
      final hash = block.dataHash;

      // scale + hashes + data in one buffer:
      // 4 bytes (f32) + 4 + 4 bytes (u32 + u32) + data (len * 2) (int16)
      final bytes = Uint8List(4 + 4 + 4 + len * 2);
      final bd = ByteData.sublistView(bytes);

      int offset = 0;

      // scale (f32)
      bd.setFloat32(offset, block.scale, Endian.little);
      offset += 4;

      // hash1
      bd.setUint32(offset, hash.$1, Endian.little);
      offset += 4;

      // hash2
      bd.setUint32(offset, hash.$2, Endian.little);
      offset += 4;

      // quantized data
      for (int i = 0; i < len; i++) {
        bd.setInt16(offset, data[i], Endian.little);
        offset += 2;
      }

      dataWriter.writeBytes(bytes);
    }
  }

  void writeTensorBF16(Float32List tensor) {
    final out = ByteData(tensor.length * 2);
    final buffer = ByteData(4);

    for (int i = 0; i < tensor.length; i++) {
      buffer.setFloat32(0, tensor[i], Endian.little);

      final o = i * 2;
      out.setUint8(o, buffer.getUint8(2));
      out.setUint8(o + 1, buffer.getUint8(3));
    }

    dataWriter.writeBytes(out.buffer.asUint8List());
  }

  void writeTensorBF16H(Float32List tensor) {
    final size = tensor.length;

    final out = ByteData(size * 2);
    final buffer = ByteData(4);

    final hash = Hash64();

    for (int i = 0; i < size; i++) {
      buffer.setFloat32(0, tensor[i], Endian.little);
      final v = buffer.getUint16(2, Endian.little);
      out.setUint16(i * 2, v, Endian.little);
      hash.add16(v);
    }

    final (h1, h2) = hash.finish();

    dataWriter.writeBytes(out.buffer.asUint8List());
    dataWriter.writeU32(h1);
    dataWriter.writeU32(h2);
  }

  void writeTensorFP32(Float32List tensor) {
    final bd = ByteData(tensor.length * 4);
    for (int i = 0; i < tensor.length; i++) {
      bd.setFloat32(i * 4, tensor[i], Endian.little);
    }
    dataWriter.writeBytes(bd.buffer.asUint8List());
  }

  Future<void> writeTensorFromRepo(
    TensorRepository repo,
    String tensorName,
    QuantType type,
  ) async {
    final values = await repo.readTensor(tensorName);
    await writeTensor(values, type);
  }
}

class HFNames {
  static String layer(int l, String suffix) {
    return 'model.layers.$l.$suffix';
  }

  /* --- Embeddings --- */
  static const String embedTokens = 'model.embed_tokens.weight';

  /* --- Final norm --- */
  static const String finalNorm = 'model.norm.weight';

  /* --- Attention --- */
  static String attnInputNorm(int l) => layer(l, 'input_layernorm.weight');

  static String qProj(int l) => layer(l, 'self_attn.q_proj.weight');

  static String kProj(int l) => layer(l, 'self_attn.k_proj.weight');

  static String vProj(int l) => layer(l, 'self_attn.v_proj.weight');

  static String oProj(int l) => layer(l, 'self_attn.o_proj.weight');

  static String attnPostNorm(int l) =>
      layer(l, 'post_attention_layernorm.weight');

  /* --- MLP --- */
  static String mlpGate(int l) => layer(l, 'mlp.gate_proj.weight');

  static String mlpUp(int l) => layer(l, 'mlp.up_proj.weight');

  static String mlpDown(int l) => layer(l, 'mlp.down_proj.weight');
}

class TensorRepositoryLoader {
  static Future<TensorRepository> load(String modelPath) async {
    final file = File(modelPath);

    // Case 1: direct safetensors file
    if (await file.exists()) {
      return SafeTensorFileRepository.load(modelPath);
    }

    final possibleIndexPaths = [
      '$modelPath.safetensors.index.json',
      '$modelPath.index.json',
    ];

    var indexPaths = await Future.wait(
      possibleIndexPaths.map((fp) async {
        var f = File(fp);
        var exists = await f.exists();
        return (f, exists);
      }),
    );

    // Case 2: sharded model index
    var indexFile = indexPaths.firstWhereOrNull((e) => e.$2)?.$1;
    if (indexFile == null) {
      throw ArgumentError('Model not found as file or shard index: $modelPath');
    }

    return SafeTensorShardRepository.load(indexFile.path);
  }
}

/* ---------------- Exporter ---------------- */

class SmolLM2Exporter {
  final HFConfig config;
  final HFTokenizer tokenizer;
  final TensorRepository repo;
  final TensorBinaryWriter writer;

  SmolLM2Exporter({
    required this.config,
    required this.tokenizer,
    required this.repo,
    required this.writer,
  });

  Future<void> export(QuantType quantType) async {
    _writeHeader(quantType);
    _writeTokenizer();
    await _writeEmbeddings(quantType);
    await _writeLayers(quantType);
    await _writeFinalNorm();
  }

  void _writeHeader(QuantType quantType) {
    final w = writer.dataWriter;

    w.writeBytes(utf8.encode('SMOL'));
    w.writeU32(2);
    w.writeU32(quantType.value);
    w.writeU32(0);

    w.writeU32(config.hiddenSize);
    w.writeU32(config.intermediateSize);
    w.writeU32(config.numHiddenLayers);
    w.writeU32(config.numAttentionHeads);
    w.writeU32(config.numKeyValueHeads);
    w.writeU32(config.vocabSize);
    w.writeU32(config.maxPositionEmbeddings);
    w.writeF32(config.ropeTheta);
    w.writeF32(config.rmsNormEps);
  }

  void _writeTokenizer() {
    final w = writer.dataWriter;

    w.writeU32(tokenizer.vocab.length);
    w.writeU32(tokenizer.merges.length);

    for (final v in tokenizer.vocab) {
      w.writeString(v);
    }

    for (final m in tokenizer.merges) {
      w.writeString(m.$1);
      w.writeString(m.$2);
    }
  }

  Future<void> _writeEmbeddings(QuantType type) async {
    await writer.dataWriter.flush();

    print('Writing embeddings...');
    await writer.writeTensorFromRepo(repo, 'model.embed_tokens.weight', type);
  }

  Future<void> _writeLayers(QuantType type) async {
    await writer.dataWriter.flush();

    for (int l = 0; l < config.numHiddenLayers; l++) {
      print('Layer $l/${config.numHiddenLayers - 1} (${type.name})');
      await _writeLayer(l, type);
    }
  }

  Future<void> _writeLayer(int l, QuantType type) async {
    final w = writer;

    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'input_layernorm.weight'),
      QuantType.fp32,
    );

    await w.writeTensorFromRepo(repo, HFNames.qProj(l), type);
    await w.writeTensorFromRepo(repo, HFNames.kProj(l), type);
    await w.writeTensorFromRepo(repo, HFNames.vProj(l), type);
    await w.writeTensorFromRepo(repo, HFNames.oProj(l), type);

    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'post_attention_layernorm.weight'),
      QuantType.fp32,
    );

    await w.writeTensorFromRepo(repo, HFNames.mlpGate(l), type);
    await w.writeTensorFromRepo(repo, HFNames.mlpUp(l), type);
    await w.writeTensorFromRepo(repo, HFNames.mlpDown(l), type);
  }

  Future<void> _writeFinalNorm() async {
    print('Writing final norm...');
    await writer.writeTensorFromRepo(repo, 'model.norm.weight', QuantType.fp32);
  }
}

/* ---------------- Entry ---------------- */

Future<void> exportSmolLM2({
  required String configPath,
  required String tokenizerPath,
  required String modelPath,
  required String outputPath,
  QuantType quantType = QuantType.q16,
}) async {
  print('=== SmolLM2 Export ===');

  print('Paths:');
  print('  config: $configPath');
  print('  tokenizer: $tokenizerPath');
  print('  model: $modelPath');
  print('  output: $outputPath');
  print('  quantize: ${quantType.name}');

  print('Loading config...');
  final config = HFConfig.fromJson(
    jsonDecode(await File(configPath).readAsString()),
  );

  print('Loading tokenizer...');
  final tokenizer = await HFTokenizer.load(tokenizerPath);

  print('Loading tensors...');
  final repo = await TensorRepositoryLoader.load(modelPath);

  final sink = File(outputPath).openWrite();
  final bw = DataWriter(sink);

  final exporter = SmolLM2Exporter(
    config: config,
    tokenizer: tokenizer,
    repo: repo,
    writer: TensorBinaryWriter(bw),
  );

  await exporter.export(quantType);

  await sink.flush();
  await sink.close();
}
