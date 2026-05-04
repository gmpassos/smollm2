import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

class DataReader {
  final RandomAccessFile file;

  DataReader(this.file);

  final _byteData = ByteData(4);
  late final _byteDataBytes = _byteData.buffer.asUint8List();

  int readU32() {
    var r = file.readIntoSync(_byteDataBytes, 0, 4);
    if (r != 4) {
      throw Exception('Unexpected EOF');
    }

    return _byteData.getUint32(0, Endian.little);
  }

  double readF32() {
    var r = file.readIntoSync(_byteDataBytes, 0, 4);
    if (r != 4) {
      throw Exception('Unexpected EOF');
    }

    return _byteData.getFloat32(0, Endian.little);
  }

  Uint8List readBytes(int n) {
    final bytes = file.readSync(n);
    if (bytes.length != n) {
      throw Exception('Read error> ${bytes.length} != $n');
    }
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

  final _byteData = ByteData(4);

  void writeU32(int v) {
    final byteData = _byteData;
    byteData.setUint32(0, v, Endian.little);
    sink.add(Uint8List.fromList(byteData.buffer.asUint8List()));
  }

  void writeF32(double v) {
    final byteData = _byteData;
    byteData.setFloat32(0, v, Endian.little);
    sink.add(Uint8List.fromList(byteData.buffer.asUint8List()));
  }

  void writeBytes(Uint8List bytes) {
    sink.add(bytes);
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

    int h1 = 2166136261; // FNV-1a
    int h2 = 2166136261 ^ 0x9e3779b9; // different offset basis

    // Hash 1: normal forward FNV-1a
    for (int i = 0; i < data.length; i++) {
      final v = data[i] & 0xFF;
      h1 ^= v;
      h1 = (h1 * 16777619) & 0xFFFFFFFF;
    }

    // Hash 2: reversed + different mixing
    for (int i = data.length - 1; i >= 0; i--) {
      int v = data[i] & 0xFF;

      // extra diffusion before mixing
      v = (v ^ (v << 5)) & 0xFF;

      h2 ^= v;
      h2 = (h2 * 2246822519) & 0xFFFFFFFF; // different prime (xxHash-inspired)
      h2 = ((h2 << 13) | (h2 >> 19)) & 0xFFFFFFFF; // rotation step
    }

    return (h1.abs(), h2.abs());
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
