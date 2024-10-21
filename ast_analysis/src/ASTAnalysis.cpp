#include "ASTAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

std::string getIndentation(int depth) {
    return std::string(depth * 2, ' ');
}

void printAST(mlir::Operation *op, int depth) {
    std::string indent = getIndentation(depth);

    std::cout << indent << "Operation: " << op->getName().getStringRef().str();

    if (op->getNumResults() > 0) {
        std::cout << " ->";
        for (auto result : op->getResults()) {
            std::cout << " ";

            std::string resultTypeStr;
            llvm::raw_string_ostream resultStream(resultTypeStr);

            result.getType().print(resultStream);

            resultStream.flush();
            std::cout << resultTypeStr;
        }
    }
    std::cout << std::endl;

    for (unsigned i = 0; i < op->getNumRegions(); ++i) {
        mlir::Region &region = op->getRegion(i);
        std::cout << indent << "  Region:" << std::endl;

        for (auto &block : region) {
            std::cout << indent << "    Block:" << std::endl;

            for (auto &innerOp : block) {
                printAST(&innerOp, depth + 3);
            }
        }
    }
}

void ASTAnalysis::analyze(mlir::ModuleOp &&module) {
    std::cout << "AST Representation of Module:\n";
    printAST(module.getOperation(), 0);
}
