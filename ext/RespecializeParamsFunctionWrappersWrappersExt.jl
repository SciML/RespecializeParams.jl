module RespecializeParamsFunctionWrappersWrappersExt

using RespecializeParams: RespecializeParams, OpaqueVoid, opaque_container_type,
    opaque_signature
import FunctionWrappersWrappers

function RespecializeParams.wrap_void_opaque(ff, ::Type{P}, sigs::Tuple) where {P}
    C = opaque_container_type(P)
    opaque_sigs = map(s -> opaque_signature(s, C), sigs)
    nothings = map(_ -> Nothing, sigs)
    return FunctionWrappersWrappers.FunctionWrappersWrapper(
        OpaqueVoid(P, ff), opaque_sigs, nothings,
    )
end

end
