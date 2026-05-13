enum QuantType {
  q8(1, true),
  q16(2, true),
  q16PerBlock(3, true),
  bf16(4, true),
  fp32(5, false);

  final int value;
  final bool quantized;

  const QuantType(this.value, this.quantized);

  factory QuantType.withValue(int value) {
    for (var e in QuantType.values) {
      if (e.value == value) {
        return e;
      }
    }
    throw ArgumentError("No `QuantType` with value: $value");
  }
}
