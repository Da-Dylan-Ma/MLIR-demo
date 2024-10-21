#ifndef AST_ANALYSIS_H
#define AST_ANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

class ASTAnalysis {
public:
    void analyze(mlir::ModuleOp &&module);
};

void printAST(mlir::Operation *op, int indentLevel);

#endif // AST_ANALYSIS_H
