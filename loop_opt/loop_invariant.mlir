module {
  func.func @scalar_multiplication(%A: memref<100xf32>, %C: memref<100xf32>) {
    affine.for %i = 0 to 100 {
      %scalar = arith.constant 2.0 : f32
      %a = affine.load %A[%i] : memref<100xf32>
      %prod = arith.mulf %a, %scalar : f32
      affine.store %prod, %C[%i] : memref<100xf32>
    }
    return
  }
}
