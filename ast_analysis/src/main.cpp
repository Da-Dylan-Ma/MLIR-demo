#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "ASTAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "Usage: " << argv[0] << " <MLIR file>\n";
    return 1;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFile(argv[1]);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Error reading input file: " << argv[1] << "\n";
    return 1;
  }

  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registerDialects(registry);
  context.appendDialectRegistry(registry);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse MLIR file: " << argv[1] << "\n";
    return 1;
  }

  ASTAnalysis analysis;
  analysis.analyze(std::move(*module));

  return 0;
}
