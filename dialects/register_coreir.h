#ifndef REGISTER_COREIR_H
#define REGISTER_COREIR_H

namespace mlir {
class MLIRContext;
}

namespace coreir {
void registerCoreIRDialect(mlir::MLIRContext *ctx);
}

#endif
