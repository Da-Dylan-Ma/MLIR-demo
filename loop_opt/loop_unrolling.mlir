module {
  func.func @vector_add(%A: memref<100xf32>, %B: memref<100xf32>, %C: memref<100xf32>) {
    affine.for %i = 0 to 100 {
      %a = affine.load %A[%i] : memref<100xf32>
      %b = affine.load %B[%i] : memref<100xf32>
      %sum = arith.addf %a, %b : f32
      affine.store %sum, %C[%i] : memref<100xf32>
    }
    return
  }
}
