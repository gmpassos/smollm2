import Foundation
import Accelerate

// ============================================================
// Matrix Vector Multiplication
// output = weights(rows x cols) * input(cols)
// ============================================================

@_cdecl("cpu_matmul")
public func cpu_matmul(
    _ weights: UnsafePointer<Float>,
    _ input: UnsafePointer<Float>,
    _ output: UnsafeMutablePointer<Float>,
    _ rows: Int32,
    _ cols: Int32
) {
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        rows,
        cols,
        1.0,
        weights,
        cols,
        input,
        1,
        0.0,
        output,
        1
    )
}

// ============================================================
// Logits Computation
// logits[vocab] = emb(vocab x hs) * x(hs)
// ============================================================

@_cdecl("cpu_compute_logits")
public func cpu_compute_logits(
    _ x: UnsafePointer<Float>,
    _ emb: UnsafePointer<Float>,
    _ logits: UnsafeMutablePointer<Float>,
    _ hs: Int32,
    _ vocabSize: Int32
) {
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        vocabSize,
        hs,
        1.0,
        emb,
        hs,
        x,
        1,
        0.0,
        logits,
        1
    )
}