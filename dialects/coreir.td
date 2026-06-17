#ifndef COREIR_DIALECT
#define COREIR_DIALECT

include "mlir/IR/OpBase.td"

def CoreIR_Dialect : Dialect {
  let name = "coreir";
  let cppNamespace = "::coreir";
  let summary = "A self-learning intermediate representation";
  let description = [{
    CoreIR provides three primitive operations for building software
    that can learn, search, and safely modify itself at runtime.
  }];
}

class CoreIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<CoreIR_Dialect, mnemonic, traits>;

def DiffOp : CoreIR_Op<"diff"> {
  let summary = "Differentiable computation";
  let arguments = (ins AnyType:$function);
  let results = (outs AnyType:$gradient);
  let assemblyFormat = "$function attr-dict";
}

def SearchOp : CoreIR_Op<"search"> {
  let summary = "Probabilistic search over program space";
  let arguments = (ins AnyType:$space, F64Attr:$temperature);
  let results = (outs AnyType:$sample);
  let assemblyFormat = "$space $temperature attr-dict";
}

def RewriteOp : CoreIR_Op<"rewrite"> {
  let summary = "Safe runtime self-modification";
  let arguments = (ins AnyType:$target, AnyType:$replacement, AnyType:$proof);
  let results = (outs I1:$success);
  let assemblyFormat = "$target $replacement $proof attr-dict";
}

#endif
