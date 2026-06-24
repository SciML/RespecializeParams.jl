using RespecializeParams
using SciMLTesting
using Aqua
using ExplicitImports
using JET
using Test

# `Base.RefValue` is not part of Base's public API, but `OpaqueRef` deliberately
# stores its payload in a `Base.RefValue{Any}` field: `Ref{Any}` (the public name)
# is an abstract type, so a `ref::Ref{Any}` field would be non-concrete and defeat
# this package's whole purpose (type-stable, fixed-layout containers). There is no
# public Base name for the concrete boxed-`Any` slot, so the lone
# all_qualified_accesses_are_public violation is ignored here, scoped to that one
# check and that one name only.
run_qa(
    RespecializeParams;
    Aqua = Aqua,
    JET = JET,
    jet = true,
    jet_kwargs = (; target_modules = (RespecializeParams,)),
    ExplicitImports = ExplicitImports,
    explicit_imports = true,
    ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:RefValue,))),
)
