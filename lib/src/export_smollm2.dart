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
  final List<String> merges;

  HFTokenizer(this.vocab, this.merges);

  static Future<HFTokenizer> load(String path) async {
    final jsonMap = jsonDecode(await File(path).readAsString());
    final model = jsonMap['model'] as Map<String, dynamic>;
    final vocabMap = model['vocab'] as Map<String, dynamic>;
    final merges = (model['merges'] as List).cast<String>();

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

  (int, int) get dataHash => _dataHash ??= data.hashListInt2();
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
  int get hashCode => Object.hash(scale, ListEquality().hash(data));

  @override
  String toString() => 'Q8Quantized{scale: $scale, data: ${data.length}}';
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
  int get hashCode => Object.hash(scale, ListEquality().hash(data));

  @override
  String toString() => 'Q16Quantized{scale: $scale, data: ${data.length}}';
}

/* ---------------- Decoders ---------------- */

abstract class TensorDTypeDecoder {
  Float32List decodeTensor(Uint8List bytes);

  double decodeScalar(Uint8List raw, int byteOffset);
}

/* F32 */
class F32Decoder implements TensorDTypeDecoder {
  @override
  Float32List decodeTensor(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);
    final out = Float32List(bytes.length >> 2);

    for (int i = 0; i < out.length; i++) {
      out[i] = bd.getFloat32(i << 2, Endian.little);
    }
    return out;
  }

  @override
  double decodeScalar(Uint8List raw, int byteOffset) {
    return ByteData.sublistView(raw).getFloat32(byteOffset, Endian.little);
  }
}

/* F16 */
class F16Decoder implements TensorDTypeDecoder {
  @override
  Float32List decodeTensor(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);
    final out = Float32List(bytes.length >> 1);

    for (int i = 0; i < out.length; i++) {
      out[i] = _decodeHalf(bd.getUint16(i << 1, Endian.little));
    }
    return out;
  }

  @override
  double decodeScalar(Uint8List raw, int byteOffset) {
    final h = ByteData.sublistView(raw).getUint16(byteOffset, Endian.little);
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
      v = double.infinity;
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
    final out = Float32List(bytes.length >> 1);

    final dataBuffer = ByteData(4);
    for (int i = 0; i < out.length; i++) {
      dataBuffer.setUint16(
        2,
        bd.getUint16(i << 1, Endian.little),
        Endian.little,
      );
      out[i] = dataBuffer.getFloat32(0, Endian.little);
    }

    return out;
  }

  @override
  double decodeScalar(Uint8List raw, int byteOffset) {
    final dataBuffer = ByteData(4);
    dataBuffer.setUint16(
      2,
      ByteData.sublistView(raw).getUint16(byteOffset, Endian.little),
      Endian.little,
    );
    return dataBuffer.getFloat32(0, Endian.little);
  }
}

/* Q16 */
class Q16Decoder implements TensorDTypeDecoder {
  @override
  Float32List decodeTensor(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);

    final scale = bd.getFloat32(0, Endian.little);
    final outLen = (bytes.length - 4) >> 1;
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
  double decodeScalar(Uint8List raw, int byteOffset) {
    final bd = ByteData.sublistView(raw);
    final scale = bd.getFloat32(0, Endian.little);
    final v = bd.getInt16(byteOffset, Endian.little);
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
    final byteSize = meta.dtype == 'F32' ? 4 : 2;
    return decoder.decodeScalar(raw, meta.begin + index * byteSize);
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
    double maxAbs = 0.0;

    var length = src.length;
    for (int i = 0; i < length; i++) {
      final v = src[i].abs();
      if (v > maxAbs) maxAbs = v;
    }

    final scale = maxAbs == 0 ? 1.0 : maxAbs / 127.0;
    final out = Int8List(length);

    for (int i = 0; i < length; i++) {
      final q = (src[i] / scale).round();
      out[i] = q.clamp(-127, 127);
    }

    return Q8Quantized(scale, out);
  }
}

class Q16Quantizer {
  Q16Quantized quantize(Float32List src) {
    double maxAbs = 0.0;

    var length = src.length;
    for (int i = 0; i < length; i++) {
      final v = src[i].abs();
      if (v > maxAbs) maxAbs = v;
    }

    final scale = maxAbs == 0 ? 1.0 : maxAbs / 32767.0;
    final out = Int16List(length);

    for (int i = 0; i < length; i++) {
      final q = (src[i] / scale).round();
      out[i] = q.clamp(-32767, 32767);
    }

    return Q16Quantized(scale, out);
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
        final q = q8.quantize(tensor);
        final dataHash = q.dataHash;
        dataWriter.writeF32(q.scale);
        dataWriter.writeU32(dataHash.$1);
        dataWriter.writeU32(dataHash.$2);
        dataWriter.writeBytes(q.data.buffer.asUint8List());
        break;

      case QuantType.q16:
        final q = q16.quantize(tensor);
        final dataHash = q.dataHash;
        dataWriter.writeF32(q.scale);
        dataWriter.writeU32(dataHash.$1);
        dataWriter.writeU32(dataHash.$2);
        final bd = ByteData(q.data.length * 2);
        for (int i = 0; i < q.data.length; i++) {
          bd.setInt16(i * 2, q.data[i], Endian.little);
        }

        dataWriter.writeBytes(bd.buffer.asUint8List());
        break;

      case QuantType.fp32:
        final bd = ByteData(tensor.length * 4);
        for (int i = 0; i < tensor.length; i++) {
          bd.setFloat32(i << 2, tensor[i], Endian.little);
        }
        dataWriter.writeBytes(bd.buffer.asUint8List());
        break;
    }

    await dataWriter.flush();
  }

  Future<void> writeTensorFromRepo(
    TensorRepository repo,
    String tensorName,
    QuantType type,
  ) async {
    final info = await repo.info(tensorName);
    final values = Float32List(info.count);

    for (int i = 0; i < info.count; i++) {
      values[i] = await repo.readScalar(tensorName, i);
    }

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

    // Case 2: sharded model index
    final indexPath = '$modelPath.index.json';
    final indexFile = File(indexPath);

    if (!await indexFile.exists()) {
      throw ArgumentError('Model not found as file or shard index: $modelPath');
    }

    return SafeTensorShardRepository.load(indexPath);
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
      w.writeString(m);
    }
  }

  Future<void> _writeEmbeddings(QuantType type) async {
    print('Writing embeddings...');
    await writer.writeTensorFromRepo(repo, 'model.embed_tokens.weight', type);
  }

  Future<void> _writeLayers(QuantType type) async {
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

    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'self_attn.q_proj.weight'),
      type,
    );
    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'self_attn.k_proj.weight'),
      type,
    );
    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'self_attn.v_proj.weight'),
      type,
    );
    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'self_attn.o_proj.weight'),
      type,
    );

    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'post_attention_layernorm.weight'),
      QuantType.fp32,
    );

    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'mlp.gate_proj.weight'),
      type,
    );
    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'mlp.up_proj.weight'),
      type,
    );
    await w.writeTensorFromRepo(
      repo,
      HFNames.layer(l, 'mlp.down_proj.weight'),
      type,
    );
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
