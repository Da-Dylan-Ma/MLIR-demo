cmake_minimum_required(VERSION 3.10)
project(ASTAnalysisDemo LANGUAGES CXX)

set(LLVM_DIR /home/mad29/projects/mlir/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR /home/mad29/projects/mlir/llvm-project/build/lib/cmake/mlir)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Add the correct include directories
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(/home/mad29/projects/mlir/llvm-project/mlir/include)

add_executable(ast_analysis_demo
    src/main.cpp
    src/ASTAnalysis.cpp
)

target_link_libraries(ast_analysis_demo
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRFunctionInterfaces
)
