enum QuantType {
  q8(1),
  q16(2),
  fp32(3);

  final int value;

  const QuantType(this.value);

  factory QuantType.withValue(int value) {
    for (var e in QuantType.values) {
      if (e.value == value) {
        return e;
      }
    }
    throw ArgumentError("No `QuantType` with value: $value");
  }
}
