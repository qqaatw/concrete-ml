#include "mlir/IR/Region.h"

#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace mlir {
namespace zamalang {
using ::mlir::zamalang::MidLFHE::AddPlainOp;
using ::mlir::zamalang::MidLFHE::GLWECipherTextType;
using ::mlir::zamalang::MidLFHE::HAddOp;
using ::mlir::zamalang::MidLFHE::MulPlainOp;

bool predPBSRegion(::mlir::Region &region) {
  if (region.getBlocks().size() != 1) {
    return false;
  }
  auto args = region.getBlocks().front().getArguments();
  if (args.size() != 1) {
    return false;
  }
  return args.front().getType().isa<mlir::IntegerType>();
}

void emitOpErrorForIncompatibleGLWEParameter(::mlir::OpState &op,
                                             ::llvm::Twine parameter) {
  ::llvm::Twine msg("should have the same GLWE ");
  op.emitError(msg.concat(parameter).concat(" parameter"));
}

bool verifyAddResultPadding(::mlir::OpState &op, GLWECipherTextType &in,
                            GLWECipherTextType &out) {
  // If the input has no value of paddingBits that doesn't constraint output.
  if (in.getPaddingBits() == -1) {
    return true;
  }
  // If the input has 0 paddingBits the ouput should have 0 paddingBits
  if (in.getPaddingBits() == 0) {
    if (out.getPaddingBits() != 0) {
      op.emitError(
          "the result shoud have 0 paddingBits has input has 0 paddingBits");
      return false;
    }
    return true;
  }
  if (in.getPaddingBits() != out.getPaddingBits() + 1) {
    op.emitError("the result should have one less padding bit than the input");
    return false;
  }
  return true;
}

bool verifyAddResultHasSameParameters(::mlir::OpState &op,
                                      GLWECipherTextType &in,
                                      GLWECipherTextType &out) {
  if (in.getDimension() != out.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return false;
  }
  if (in.getPolynomialSize() != out.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return false;
  }
  if (in.getBits() != out.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return false;
  }
  if (in.getP() != out.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return false;
  }
  if (in.getPhantomBits() != out.getPhantomBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "phantomBits");
    return false;
  }
  if (in.getScalingFactor() != out.getScalingFactor()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "scalingFactor");
    return false;
  }
  if (in.getLog2StdDev() != out.getLog2StdDev()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "log2StdDev");
    return false;
  }
  return true;
}

::mlir::LogicalResult verifyAddPlainOp(AddPlainOp &op) {
  GLWECipherTextType in = op.a().getType().cast<GLWECipherTextType>();
  GLWECipherTextType out = op.getResult().getType().cast<GLWECipherTextType>();
  if (!verifyAddResultPadding(op, in, out)) {
    return ::mlir::failure();
  }
  if (!verifyAddResultHasSameParameters(op, in, out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

bool verifyHAddResultPadding(::mlir::OpState &op, GLWECipherTextType &inA,
                             GLWECipherTextType &inB, GLWECipherTextType &out) {
  // If the inputs has no value of paddingBits that doesn't constraint output.
  if (inA.getPaddingBits() == -1 && inB.getPaddingBits() == -1) {
    return true;
  }
  return verifyAddResultPadding(op, inA, out);
}

int hAddLog2StdDevOfResult(int a, int b) {
  long double va = std::pow(std::pow(2, a), 2);
  long double vb = std::pow(std::pow(2, b), 2);
  long double vr = va + vb;
  return std::log2(::sqrt(vr));
}

bool verifyHAddResultLog2StdDev(::mlir::OpState &op, GLWECipherTextType &inA,
                                GLWECipherTextType &inB,
                                GLWECipherTextType &out) {
  // If the inputs has no value of log2StdDev that doesn't constraint output.
  if (inA.getLog2StdDev() == -1 && inB.getLog2StdDev() == -1) {
    return true;
  }
  int expectedLog2StdDev =
      hAddLog2StdDevOfResult(inA.getLog2StdDev(), inB.getLog2StdDev());
  if (out.getLog2StdDev() != expectedLog2StdDev) {
    ::llvm::Twine msg(
        "has unexpected log2StdDev parameter of its GLWE result, expected:");
    op.emitOpError(msg.concat(::llvm::Twine(expectedLog2StdDev)));
    return false;
  }
  return true;
}

bool verifyHAddSameGLWEParameter(::mlir::OpState &op, GLWECipherTextType &inA,
                                 GLWECipherTextType &inB,
                                 GLWECipherTextType &out) {
  if (inA.getDimension() != inB.getDimension() ||
      inA.getDimension() != out.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return false;
  }
  if (inA.getPolynomialSize() != inB.getPolynomialSize() &&
      inA.getPolynomialSize() != out.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return false;
  }
  if (inA.getBits() != inB.getBits() && inA.getBits() != out.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return false;
  }
  if (inA.getP() != inB.getP() && inA.getP() != out.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return false;
  }
  if (inA.getPhantomBits() != inB.getPhantomBits() &&
      inA.getPhantomBits() != out.getPhantomBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "phantomBits");
    return false;
  }
  if (inA.getScalingFactor() && inB.getScalingFactor() &&
      inA.getScalingFactor() != out.getScalingFactor()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "scalingFactor");
    return false;
  }
  return true;
}

::mlir::LogicalResult verifyHAddOp(HAddOp &op) {
  GLWECipherTextType inA = op.a().getType().cast<GLWECipherTextType>();
  GLWECipherTextType inB = op.b().getType().cast<GLWECipherTextType>();
  GLWECipherTextType out = op.getResult().getType().cast<GLWECipherTextType>();
  if (!verifyHAddResultPadding(op, inA, inB, out)) {
    return ::mlir::failure();
  }
  if (!verifyHAddResultLog2StdDev(op, inA, inB, out)) {
    return ::mlir::failure();
  }
  if (!verifyHAddSameGLWEParameter(op, inA, inB, out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

bool verifyMulPlainOpPadding(::mlir::OpState &op, GLWECipherTextType &inA,
                             ::mlir::Value &inB, GLWECipherTextType &out) {
  if (inA.getPaddingBits() == -1) {
    return true;
  }
  if (inA.getPaddingBits() == 0) {
    if (out.getPaddingBits() != 0) {
      op.emitError(
          "the result shoud have 0 paddingBits has input has 0 paddingBits");
      return false;
    }
    return true;
  }
  unsigned int additionalBit = 0;
  ::mlir::ConstantIntOp constantOp = inB.getDefiningOp<::mlir::ConstantIntOp>();
  if (constantOp != nullptr) {
    int64_t value = constantOp.getValue();
    additionalBit = std::ceil(std::log2(value)) + 1;
  } else {
    ::mlir::IntegerType tyB = inB.getType().cast<::mlir::IntegerType>();
    additionalBit = tyB.getIntOrFloatBitWidth();
  }
  unsigned int expectedPadding = inA.getPaddingBits() - additionalBit;
  if (out.getPaddingBits() != expectedPadding) {
    ::llvm::Twine msg(
        "has unexpected padding parameter of its GLWE result, expected:");
    op.emitOpError(msg.concat(::llvm::Twine(expectedPadding)));
    return false;
  }
  return true;
}

bool verifyMulPlainResultHasSameParameters(::mlir::OpState &op,
                                           GLWECipherTextType &in,
                                           GLWECipherTextType &out) {
  if (in.getDimension() != out.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return false;
  }
  if (in.getPolynomialSize() != out.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return false;
  }
  if (in.getBits() != out.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return false;
  }
  if (in.getP() != out.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return false;
  }
  if (in.getPhantomBits() != out.getPhantomBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "phantomBits");
    return false;
  }
  if (in.getScalingFactor() != out.getScalingFactor()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "scalingFactor");
    return false;
  }
  if (in.getLog2StdDev() != out.getLog2StdDev()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "log2StdDev");
    return false;
  }
  return true;
}

::mlir::LogicalResult verifyMulPlainOp(MulPlainOp &op) {
  GLWECipherTextType inA = op.a().getType().cast<GLWECipherTextType>();
  ::mlir::Value inB = op.b();
  GLWECipherTextType out = op.getResult().getType().cast<GLWECipherTextType>();
  if (!verifyMulPlainOpPadding(op, inA, inB, out)) {
    return ::mlir::failure();
  }
  if (!verifyMulPlainResultHasSameParameters(op, inA, out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.cpp.inc"
