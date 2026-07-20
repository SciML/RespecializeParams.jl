using RespecializeParams
using SciMLTesting
using JET
using Test

# `all_qualified_accesses_are_public` ignore-list, scoped to this one check and
# these three names only:
#  * `Base.RefValue` is not part of Base's public API, but `OpaqueRef` deliberately
#    stores its payload in a `Base.RefValue{Any}` field: `Ref{Any}` (the public name)
#    is an abstract type, so a `ref::Ref{Any}` field would be non-concrete and defeat
#    this package's whole purpose (type-stable, fixed-layout containers). There is no
#    public Base name for the concrete boxed-`Any` slot.
#  * `Base.GC.@preserve` and `Base.unsafe_convert` ARE public Base API on Julia 1.11+
#    (`Base.ispublic` returns `true` for both), and the package uses them correctly.
#    But the public-API metadata mechanism did not exist before 1.11 (`Base.ispublic`
#    is undefined on 1.10), so on the LTS lane ExplicitImports cannot see that they
#    are public and false-positives them. They are ignored to keep the LTS lane green
#    without hiding any real non-public access.
run_qa(
    RespecializeParams;
    jet_kwargs = (; target_modules = (RespecializeParams,)),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (:RefValue, Symbol("@preserve"), :unsafe_convert),
        ),
    ),
)
