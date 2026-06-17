#include "coreir_dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace coreir;

//===----------------------------------------------------------------------===//
// CoreIRDialect
//===----------------------------------------------------------------------===//

CoreIRDialect::CoreIRDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<CoreIRDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "coreir/CoreIRDialect.cpp.inc"
      >();
}

void CoreIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "coreir/CoreIRDialect.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DiffOp
//===----------------------------------------------------------------------===//

LogicalResult DiffOp::verify() {
  // Verify the function input produces a valid gradient
  // In Phase 1, accept any type — full type checking comes later
  return success();
}

void DiffOp::build(OpBuilder &builder, OperationState &state, Value function) {
  state.addOperands(function);
  state.addTypes(function.getType()); // Gradient matches function type
}

//===----------------------------------------------------------------------===//
// SearchOp
//===----------------------------------------------------------------------===//

LogicalResult SearchOp::verify() {
  // Verify temperature is positive
  if (getTemperature().getValueAsDouble() <= 0.0) {
    return emitOpError("temperature must be positive");
  }
  return success();
}

void SearchOp::build(OpBuilder &builder, OperationState &state,
                     Value space, double temperature) {
  state.addOperands(space);
  state.addAttribute("temperature", builder.getF64FloatAttr(temperature));
  state.addTypes(space.getType()); // Sample matches space type
}

//===----------------------------------------------------------------------===//
// RewriteOp
//===----------------------------------------------------------------------===//

LogicalResult RewriteOp::verify() {
  // In Phase 1, basic structural check — target and replacement
  // must have compatible types
  if (getTarget().getType() != getReplacement().getType()) {
    return emitOpError("target and replacement must have same type");
  }
  return success();
}

void RewriteOp::build(OpBuilder &builder, OperationState &state,
                      Value target, Value replacement, Value proof) {
  state.addOperands({target, replacement, proof});
  state.addTypes(builder.getI1Type()); // Returns success boolean
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

// Include auto-generated dialect definitions
#include "coreir/CoreIRDialect.cpp.inc"

void coreir::registerCoreIRDialect(MLIRContext *ctx) {
  ctx->loadDialect<CoreIRDialect>();
}
