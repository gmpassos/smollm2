import 'dart:math' as math;
import 'dart:typed_data';

@pragma('vm:prefer-inline')
double silu(double x) {
  return x / (1.0 + math.exp(-x));
}

void softmax(Float32List x, int n) {
  double maxVal = x[0];

  for (int i = 1; i < n; i++) {
    if (x[i] > maxVal) {
      maxVal = x[i];
    }
  }

  double sum = 0.0;

  for (int i = 0; i < n; i++) {
    final e = math.exp(x[i] - maxVal).toDouble();
    x[i] = e;
    sum += e;
  }

  for (int i = 0; i < n; i++) {
    x[i] /= sum;
  }
}

void softmaxSegment(Float32List x, int offset, int n) {
  double maxVal = x[offset];

  for (int i = 1; i < n; i++) {
    final v = x[offset + i];
    if (v > maxVal) {
      maxVal = v;
    }
  }

  double sum = 0.0;

  for (int i = 0; i < n; i++) {
    final e = math.exp(x[offset + i] - maxVal).toDouble();
    x[offset + i] = e;
    sum += e;
  }

  for (int i = 0; i < n; i++) {
    x[offset + i] /= sum;
  }
}
