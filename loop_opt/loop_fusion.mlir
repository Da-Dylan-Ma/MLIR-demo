module {
  // Define a function named 'producer_consumer_fusion'
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    
    // Allocate two memrefs of size 10 to hold intermediate results
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.alloc() : memref<10xf32>
    
    // Define a constant value of 0.0 of type f32
    %cst = arith.constant 0.0 : f32

    // First loop: Initialize both memrefs %0 and %1 to the constant value (0.0)
    affine.for %i = 0 to 10 {
      affine.store %cst, %0[%i] : memref<10xf32>
      affine.store %cst, %1[%i] : memref<10xf32>
    }

    // Second loop: Load values from %0, double them, and store the result in %arg0
    affine.for %i = 0 to 10 {
      %val = affine.load %0[%i] : memref<10xf32>
      %doubled_val = arith.addf %val, %val : f32
      affine.store %doubled_val, %arg0[%i] : memref<10xf32>
    }

    // Third loop: Load values from %1, square them, and store the result in %arg1
    affine.for %i = 0 to 10 {
      %val = affine.load %1[%i] : memref<10xf32>
      %squared_val = arith.mulf %val, %val : f32
      affine.store %squared_val, %arg1[%i] : memref<10xf32>
    }
    
    return
  }
}
