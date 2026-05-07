import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

class DataReader {
  final RandomAccessFile file;

  DataReader(this.file);

  int _readCount = 0;
  int get readCount => _readCount;

  final _byteData = ByteData(4);
  late final _byteDataBytes = _byteData.buffer.asUint8List();

  int readU16() {
    var r = file.readIntoSync(_byteDataBytes, 0, 2);
    if (r != 2) {
      throw Exception('Unexpected EOF');
    }
    _readCount += 2;
    return _byteData.getUint16(0, Endian.little);
  }

  int readU32() {
    var r = file.readIntoSync(_byteDataBytes, 0, 4);
    if (r != 4) {
      throw Exception('Unexpected EOF');
    }
    _readCount += 4;
    return _byteData.getUint32(0, Endian.little);
  }

  double readF32() {
    var r = file.readIntoSync(_byteDataBytes, 0, 4);
    if (r != 4) {
      throw Exception('Unexpected EOF');
    }
    _readCount += 4;
    return _byteData.getFloat32(0, Endian.little);
  }

  Uint8List readBytes(int n) {
    final bytes = file.readSync(n);
    if (bytes.length != n) {
      throw Exception('Read error> ${bytes.length} != $n');
    }
    _readCount += n;
    return bytes;
  }

  String readString() {
    final len = readU32();
    return utf8.decode(readBytes(len));
  }
}

class DataWriter {
  final IOSink sink;

  DataWriter(this.sink);

  int _writeCount = 0;

  int get writeCount => _writeCount;

  final _byteData2 = ByteData(2);
  final _byteData4 = ByteData(4);

  Future<void> writeU16(int v) async {
    final byteData = _byteData2;
    byteData.setUint16(0, v, Endian.little);
    var bs = Uint8List.fromList(byteData.buffer.asUint8List());
    sink.add(bs);
    _writeCount += 2;
  }

  void writeU32(int v) {
    final byteData = _byteData4;
    byteData.setUint32(0, v, Endian.little);
    sink.add(Uint8List.fromList(byteData.buffer.asUint8List()));
    _writeCount += 4;
  }

  void writeF32(double v) {
    final byteData = _byteData4;
    byteData.setFloat32(0, v, Endian.little);
    sink.add(Uint8List.fromList(byteData.buffer.asUint8List()));
    _writeCount += 4;
  }

  void writeBytes(Uint8List bytes) {
    sink.add(bytes);
    _writeCount += bytes.length;
  }

  void writeString(String s) {
    final bytes = utf8.encode(s);
    writeU32(bytes.length);
    writeBytes(bytes);
  }

  Future<void> flush() => sink.flush();
}

typedef Float32ListX4 = ({Float32List list, Float32x4List listX4});

extension Float32ListExtension on Float32List {
  Float32x4List get asFloat32x4List => buffer.asFloat32x4List();

  Float32ListX4 get asFloat32ListX4 => (list: this, listX4: asFloat32x4List);
}

extension ListIntExtension on List<int> {
  int hashListInt() {
    final data = this;
    int hash = 2166136261;

    for (int i = 0; i < data.length; i++) {
      final v = data[i];
      hash ^= v & 0xFFFFFFFF;
      hash *= 16777619;
      hash &= 0xFFFFFFFF;
    }

    return hash.abs();
  }

  (int, int) hashListInt2() {
    final data = this;
    final len = data.length;

    int h1 = 2166136261;
    int h2 = 2166136261 ^ 0x9e3779b9;

    const fnvPrime = 16777619;
    const mul2 = 2246822519;

    for (int i = 0, j = len - 1; i < len; i++, j--) {
      final f = data[i];
      final r = data[j];

      // forward FNV-1a
      h1 ^= f;
      h1 *= fnvPrime;

      // reverse mix
      final v = (r ^ (r << 5)) & 0xFF;
      h2 ^= v;
      h2 *= mul2;
      h2 = (h2 << 13) | (h2 >>> 19);
    }

    return (h1 & 0xFFFFFFFF, h2 & 0xFFFFFFFF);
  }
}

extension MapHistogramExtension<K> on Map<K, int> {
  int count(K key) => this[key] ?? 0;

  int increment(K key, [int amount = 1]) {
    var c = count(key);
    var c2 = c + amount;
    this[key] = c2;
    return c2;
  }
}

extension RandomExtension on Random {
  Float32x4 nextFloat32x4() {
    return Float32x4(nextDouble(), nextDouble(), nextDouble(), nextDouble());
  }

  Float32x4 nextFloat32x4Scaled(double scale) {
    return Float32x4(
      nextDouble() * scale,
      nextDouble() * scale,
      nextDouble() * scale,
      nextDouble() * scale,
    );
  }

  Float32x4 nextFloat32x4Noise(double scale) {
    var d = 1 - scale;
    var noise = scale * 2;
    return Float32x4(
      d + (nextDouble() * noise),
      d + (nextDouble() * noise),
      d + (nextDouble() * noise),
      d + (nextDouble() * noise),
    );
  }

  double nextDoubleJitter(double scale) {
    var d = 1 - scale;
    var noise = scale * 2;
    return d + (nextDouble() * noise);
  }
}

class Hash64 {
  int h1 = 2166136261;
  int h2 = 2166136261;

  void add16(int v) {
    h1 = (h1 ^ v) * 16777619 & 0xFFFFFFFF;
    h2 = (h2 ^ (v >>> 1)) * 2246822519 & 0xFFFFFFFF;
  }

  (int, int) finish() => (h1, h2);

  void add32(int v) {
    add16(v & 0xFFFF);
    add16((v >>> 16) & 0xFFFF);
  }
}

extension DurationFormatting on Duration {
  String get formattedToHumanReadable {
    if (inMicroseconds < 1000) {
      return '$inMicrosecondsµs';
    }

    if (inMilliseconds < 1000) {
      return '${(inMicroseconds / 1000.0).toStringAsFixed(3)}ms';
    }

    if (inSeconds < 60) {
      return '${(inMilliseconds / 1000.0).toStringAsFixed(3)}s';
    }

    return '${(inSeconds / 60.0).toStringAsFixed(2)}min';
  }

  String get formattedToSeconds {
    var seconds = inMicroseconds / Duration.microsecondsPerSecond;
    var formatted = seconds.toStringAsFixed(3);
    return '${formatted}s';
  }
}
