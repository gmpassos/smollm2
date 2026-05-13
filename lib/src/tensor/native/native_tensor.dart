import 'dart:ffi';
import 'dart:io';

import '../../data.dart';
import '../../quant_type.dart';
import '../tensor.dart';
import 'macos/cpu/cpu_ffi.dart';
import 'macos/metal/metal_ffi.dart';
import 'windows/cuda/cuda_ffi.dart';

class NativeTensorFactory implements TensorFactory {
  static final instance = NativeTensorFactory._();

  NativeTensorFactory._();

  final Map<QuantType, List<NativeTensorReader>> _tensorReaders = {};

  @override
  NativeTensorReader<NativeTensor>? getTensorReader(QuantType dataQuantType) {
    var list = _tensorReaders[dataQuantType];
    if (list == null || list.isEmpty) {
      return null;
    }

    var tensorReader = list.first;
    return tensorReader;
  }

  @override
  Tensor readTensor(
    QuantType dataQuantType,
    DataReader dataReader,
    int rows,
    int cols, {
    bool asFP32 = false,
  }) {
    var tensorReader =
        getTensorReader(dataQuantType) ??
        (throw UnsupportedError(
          "Unsupported native tensor `QuantType`: ${dataQuantType.name}",
        ));
    return tensorReader.readTensor(dataQuantType, dataReader, rows, cols);
  }

  @override
  void load() {
    var libraries = this.libraries;

    var tensorReaders = libraries.expand((l) => l.tensorReaders).toList();

    tensorReaders.sort();
    tensorReaders = tensorReaders.toSet().toList();

    for (var tr in tensorReaders) {
      for (var q in tr.supportedQuantTypes) {
        var list = _tensorReaders[q] ??= [];
        list.add(tr);
      }
    }

    for (var l in _tensorReaders.values) {
      l.sort();
    }
  }

  List<NativeLibrary>? _libraries;

  List<NativeLibrary> get libraries =>
      _libraries ??= List.unmodifiable(listLibraries());

  List<NativeLibrary> listLibraries() {
    if (Platform.isMacOS) {
      return _listLibrariesMacOS();
    } else if (Platform.isWindows) {
      return _listLibrariesWindows();
    } else {
      return [];
    }
  }

  List<NativeLibrary> _listLibrariesMacOS() {
    var libraries = <NativeLibrary>[];

    var cpuLibrary = CPULibrary.instance;
    if (cpuLibrary.loadLibrary()) {
      libraries.add(cpuLibrary);
    }

    var metalLibrary = MetalLibrary.instance;
    if (metalLibrary.loadLibrary()) {
      libraries.add(metalLibrary);
    }

    return libraries;
  }

  List<NativeLibrary> _listLibrariesWindows() {
    final libraries = <NativeLibrary>[];

    final cudaLibrary = CudaLibrary.instance;
    if (cudaLibrary.loadLibrary()) {
      libraries.add(cudaLibrary);
    }

    return libraries;
  }
}

abstract class NativeTensor implements Tensor {}

abstract class NativeTensorReader<T extends NativeTensor>
    extends TensorReader<T> {
  Set<QuantType> get supportedQuantTypes;
}

typedef LoadLibraryResult = ({bool ok, Object? error, StackTrace? stackTrace});

class NativeBinding {}

abstract class NativeLibrary<B extends NativeBinding> {
  String get name;

  Set<NativeTensorReader> get tensorReaders;

  File? tryResolveLibraryFile();

  File resolveLibraryFile() {
    var file = tryResolveLibraryFile();
    if (file == null) {
      throw StateError(
        "Can't resolve native library file for `$name` support!",
      );
    }

    if (!file.existsSync()) {
      throw StateError(
        "Resolved native library file for `$name` does NOT exists: ${file.path}",
      );
    }

    return file;
  }

  B create();

  void loadLibraryOrThrow() {
    var loaded = loadLibraryFrom(resolveLibraryFile());
    if (!loaded) {
      throw StateError('ERROR Loading `$runtimeType` library!');
    }
  }

  LoadLibraryResult tryLoadLibrary() =>
      tryLoadLibraryFrom(resolveLibraryFile());

  bool loadLibrary() {
    var load = tryLoadLibrary();
    if (!load.ok) {
      print('ERROR Loading `$runtimeType` library> ${load.error}');
      print(load.stackTrace);
    }
    return load.ok;
  }

  bool loadLibraryFrom(File file) {
    var load = tryLoadLibraryFrom(file);
    if (load.ok) return true;

    print("ERROR: ${load.error}");
    print(load.stackTrace);
    return false;
  }

  LoadLibraryResult? _loadLibraryResult;

  bool get isLibraryLoaded => _loadLibraryResult?.ok ?? false;

  LoadLibraryResult tryLoadLibraryFrom(File file) {
    if (_loadLibraryResult != null) return _loadLibraryResult!;

    file = file.absolute;

    try {
      final path = file.path;
      DynamicLibrary.open(path);
      return _loadLibraryResult = (ok: true, error: null, stackTrace: null);
    } catch (e, s) {
      return _loadLibraryResult = (ok: false, error: e, stackTrace: s);
    }
  }
}
