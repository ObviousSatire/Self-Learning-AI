#ifndef COREIR_DIALECT_H
#define COREIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace coreir {

class CoreIRDialect : public mlir::Dialect {
public:
  explicit CoreIRDialect(mlir::MLIRContext *ctx);
  
  static llvm::StringRef getDialectNamespace() { return "coreir"; }
  
  void initialize();
};

} // namespace coreir

// Include the auto-generated op declarations
#define GET_OP_CLASSES
#include "coreir/CoreIRDialect.h.inc"

#endif
