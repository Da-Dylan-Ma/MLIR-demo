//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "toy/Dialect.h"
using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

/* Added by Da - Start */

// Helper function to check if reordering the matrices results in better performance.
// The function compares the dimensions of matrices to decide whether Ax(BC) is better than (AB)xC.
bool is_better_reordered(mlir::RankedTensorType a_type, mlir::RankedTensorType b_type, mlir::RankedTensorType c_type) {
  // Extract the dimensions of matrices A, B, and C.
  int64_t a_rows = a_type.getShape()[0];
  int64_t a_cols = a_type.getShape()[1];
  int64_t b_cols = b_type.getShape()[1];
  int64_t c_cols = c_type.getShape()[1];

  // Cost of (AB)xC: a_rows * a_cols * c_cols + a_rows * b_cols * c_cols
  int64_t cost_ab_c = a_rows * a_cols * c_cols + a_rows * b_cols * c_cols;

  // Cost of A(BC): a_rows * b_cols * c_cols + a_cols * b_cols * c_cols
  int64_t cost_a_bc = a_rows * b_cols * c_cols + a_cols * b_cols * c_cols;

  // Return true if reordering is beneficial, i.e., A(BC) is cheaper than (AB)C.
  return cost_a_bc < cost_ab_c;
}

struct OptimizeChainMatmul : public mlir::OpRewritePattern<MatmulOp> {
  OptimizeChainMatmul(mlir::MLIRContext *context)
    : OpRewritePattern<MatmulOp>(context, 2) {}

  llvm::LogicalResult
  matchAndRewrite(MatmulOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Value matmul_lhs = op.getOperands()[0];
    mlir::Value matmul_rhs = op.getOperands()[1];
    MatmulOp matmul_lhs_op = matmul_lhs.getDefiningOp<MatmulOp>();

    if (!matmul_lhs_op) return failure();

    auto lhs_a_type = matmul_lhs_op.getOperands()[0].getType();
    auto lhs_b_type = matmul_lhs_op.getOperands()[1].getType();
    auto rhs_c_type = matmul_rhs.getType();

    if (!lhs_a_type.isa<mlir::RankedTensorType>() ||
        !lhs_b_type.isa<mlir::RankedTensorType>() ||
        !rhs_c_type.isa<mlir::RankedTensorType>()) {
      return failure();
    }

    auto lhs_a = lhs_a_type.cast<mlir::RankedTensorType>();
    auto lhs_b = lhs_b_type.cast<mlir::RankedTensorType>();
    auto rhs_c = rhs_c_type.cast<mlir::RankedTensorType>();

    // Check if reordering improves performance using the helper function.
    if (!is_better_reordered(lhs_a, lhs_b, rhs_c)) {
      return failure();
    }

    // If reordering is beneficial, proceed with the optimization.
    auto bx_c = rewriter.create<MatmulOp>(op.getLoc(), matmul_lhs_op.getOperands()[1], matmul_rhs);
    auto ax_bc = rewriter.create<MatmulOp>(op.getLoc(), matmul_lhs_op.getOperands()[0], bx_c);

    // Replace the original op with the optimized Ax(BC) chain.
    rewriter.replaceOp(op, ax_bc.getResult());
    return success();
  }
};

void MatmulOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context){
  results.add<OptimizeChainMatmul>(context);
}

/* Added by Da - End */


/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
