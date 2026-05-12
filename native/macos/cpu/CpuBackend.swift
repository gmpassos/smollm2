import Foundation
import Accelerate

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