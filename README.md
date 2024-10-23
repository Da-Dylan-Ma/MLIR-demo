# MLIR Demo Project

This project demonstrates various MLIR capabilities, including loop optimization with affine dialects, a simple AST printer, and an end-to-end demo using a custom `matmul` dialect with optimizations.

## Prerequisites

Before running the demos, ensure you have MLIR installed:

- Follow the installation guide at [MLIR Getting Started](https://mlir.llvm.org/getting_started/).
- These demos were tested on an Ubuntu 24 machine, but they should work on most Linux environments.

## 1. Loop Optimization with Affine Dialect

This section showcases loop optimizations using MLIR's affine dialect.

### Instructions:
1. Navigate to the `loop_opt` directory.
2. Follow the commands listed in `cmd.txt` to observe differences between optimized and non-optimized sample codes in MLIR Intermediate Representation (IR).

## 2. Simple AST Printer

This demo processes `.mlir` files to provide a more structured and readable representation of MLIR code.

### Instructions:
- Use the provided scripts to run the AST printer on `.mlir` files, generating a clearer view of the code's structure.

## 3. End-to-End Demo with `matmul` Dialect

This section presents an end-to-end workflow using a custom `matmul` dialect, demonstrating optimizations and code generation.

### Setup Instructions:
1. Replace the contents of the `llvm-project/mlir/examples/toy/Ch6/` directory with the modified `Ch6` directory provided.
2. Ensure a complete build of the MLIR library has been completed. Then, navigate to `llvm-project/build/` and run:
   ```bash
   cmake --build .
   ```
3. Make sure to rebuild the project following MLIR's official guidelines.

### Modified Files:
Below is a list of modified files. Search for "Added by Da" in these files to quickly locate the changes:

#### Step 1:
- `llvm-project/mlir/examples/toy/Ch6/include/toy/Ops.td`
- `llvm-project/mlir/examples/toy/Ch6/mlir/MLIRGen.cpp`
- `llvm-project/mlir/examples/toy/Ch6/mlir/Dialect.cpp`

#### Step 2:
- `llvm-project/mlir/examples/toy/Ch6/include/toy/Ops.td`
- `llvm-project/mlir/examples/toy/Ch6/mlir/ToyCombine.cpp`

#### Step 3:
- `llvm-project/mlir/examples/toy/Ch6/mlir/Dialect.cpp`
- `llvm-project/mlir/examples/toy/Ch6/include/toy/Ops.td`

#### Step 4:
- `llvm-project/mlir/examples/toy/Ch6/mlir/LowerToAffineLoops.cpp`

### Commands for Each Step:

Navigate to `llvm-project/build/` and execute the following commands for each step:

#### **Step 1 Commands:**
```bash
./bin/toyc-ch5 ../llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir
```

#### **Step 2 Commands:**
```bash
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt
```

#### **Step 3 Commands:**
```bash
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt
```

#### **Step 4 Commands:**
```bash
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch5/codegen.toy -emit=mlir-affine
```

#### **Step 5 Commands:**
```bash
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch6/jit.toy -emit=mlir
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch6/jit.toy -emit=mlir-affine
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch6/jit.toy -emit=mlir-llvm
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch6/jit.toy -emit=llvm
./bin/toyc-ch6 ../llvm-project/mlir/test/Examples/Toy/Ch6/jit.toy -emit=jit
```
