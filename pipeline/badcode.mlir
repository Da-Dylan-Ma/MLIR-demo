module {
  func.func @inefficient_tensor_contraction(%A: memref<64x64x64xf32>, %B: memref<64x64x64xf32>, %C: memref<64x64x64xf32>) {
    // Invariant code inside the loop (this constant computation should be hoisted)
    %inv_const = arith.constant 2.0 : f32

    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        // Dead code that serves no purpose
        %dead1 = arith.addf %inv_const, %inv_const : f32

        affine.for %k = 0 to 64 {
          // Removed redundant memory loads and stores
          %a1 = affine.load %A[%i, %j, %k] : memref<64x64x64xf32>
          %b1 = affine.load %B[%i, %j, %k] : memref<64x64x64xf32>

          // Redundant computation (common subexpression)
          %mul1 = arith.mulf %a1, %b1 : f32
          %mul2 = arith.mulf %a1, %b1 : f32 // Same as above, redundant

          // Useless code (dead code)
          %dead2 = arith.addf %mul1, %mul1 : f32

          // Proper computation
          %c = affine.load %C[%i, %j, %k] : memref<64x64x64xf32>
          %add = arith.addf %mul1, %c : f32

          // Removed redundant store
          affine.store %add, %C[%i, %j, %k] : memref<64x64x64xf32>
        }
      }
    }
    return
  }
}
