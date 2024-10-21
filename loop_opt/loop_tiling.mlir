module {
  func.func @matrix_addition(%A: memref<100x100xf32>, %B: memref<100x100xf32>, %C: memref<100x100xf32>) {
    affine.for %i = 0 to 100 {
      affine.for %j = 0 to 100 {
        %a = affine.load %A[%i, %j] : memref<100x100xf32>
        %b = affine.load %B[%i, %j] : memref<100x100xf32>
        %sum = arith.addf %a, %b : f32
        affine.store %sum, %C[%i, %j] : memref<100x100xf32>
      }
    }
    return
  }
}
