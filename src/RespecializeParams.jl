module RespecializeParams

export OpaqueParams, OpaqueRef, OpaqueVoid,
    pack, pack_any, unpack, unsafe_unpack, unpack_checked, repack!

"""
    OpaqueParams

Type-stable container for an `isbits` value. The wrapper type is fixed regardless
of the underlying payload type, so callers (e.g. ODE/Nonlinear solvers) that
dispatch on `typeof(p)` always hit the same precompiled code path. The payload
type is recovered inside the user-supplied callback via [`unpack`](@ref), which
reinterprets the bytes back to the concrete type with no allocation.

Fields:
- `bytes::Vector{UInt8}`  — the raw payload (always heap-allocated, fixed length per instance).
- `typeid::UInt`           — `objectid` of the concrete type used at `pack` time, for
                              optional consistency checks. Not used by `unpack`.

`OpaqueParams` is intentionally **not** parametric: that's the whole point. If you
need a parametric wrapper (e.g. for dispatch in your own callback hierarchy) just
wrap it: `struct MyParams{T}; op::OpaqueParams; end`.
"""
struct OpaqueParams
    bytes::Vector{UInt8}
    typeid::UInt
end

@inline _check_isbits(::Type{T}) where {T} =
    isbitstype(T) || throw(ArgumentError("RespecializeParams requires isbitstype; got $T"))

"""
    pack(p) -> OpaqueParams

Copy the bytes of an `isbits` value `p` into a fresh `OpaqueParams`. Allocates one
`Vector{UInt8}` of length `sizeof(typeof(p))` plus the wrapper.
"""
function pack(p::T) where {T}
    _check_isbits(T)
    n = sizeof(T)
    bytes = Vector{UInt8}(undef, n)
    r = Ref(p)
    GC.@preserve r bytes begin
        dst = Ptr{T}(pointer(bytes))
        src = Base.unsafe_convert(Ptr{T}, r)
        unsafe_store!(dst, unsafe_load(src))
    end
    return OpaqueParams(bytes, objectid(T))
end

"""
    unpack(op::OpaqueParams, ::Type{T}) -> T

Reinterpret the stored bytes as a `T`. Performs a size check (cheap, branch-only,
no allocation). Type-stable: the return type is `T`. The result lives on the
stack when used in a typical numerical kernel.
"""
@inline function unpack(op::OpaqueParams, ::Type{T}) where {T}
    _check_isbits(T)
    sizeof(T) == length(op.bytes) || throw(
        ArgumentError(
            "size mismatch: $T is $(sizeof(T)) bytes, OpaqueParams holds $(length(op.bytes))"
        )
    )
    return unsafe_unpack(op, T)
end

"""
    unsafe_unpack(op::OpaqueParams, ::Type{T}) -> T

Same as `unpack` but skips the size check. Use only when `T` is known to match
what was packed. Marked `@inline`; expands to a single `unsafe_load`.
"""
@inline function unsafe_unpack(op::OpaqueParams, ::Type{T}) where {T}
    GC.@preserve op begin
        return unsafe_load(Ptr{T}(pointer(op.bytes)))
    end
end

"""
    unpack_checked(op::OpaqueParams, ::Type{T}) -> T

Like `unpack` but additionally verifies that `T` matches the type used at `pack`
time via `objectid`. Slightly more defensive; same big-O cost.
"""
@inline function unpack_checked(op::OpaqueParams, ::Type{T}) where {T}
    objectid(T) == op.typeid || throw(
        ArgumentError(
            "typeid mismatch: OpaqueParams was packed with a different concrete type"
        )
    )
    return unpack(op, T)
end

"""
    repack!(op::OpaqueParams, p::T) -> op

Overwrite the bytes of `op` in place with a new value of the same type. Useful
when a solver wants to mutate parameters between calls without re-allocating the
container.
"""
function repack!(op::OpaqueParams, p::T) where {T}
    _check_isbits(T)
    sizeof(T) == length(op.bytes) || throw(
        ArgumentError(
            "size mismatch: $T is $(sizeof(T)) bytes, OpaqueParams holds $(length(op.bytes))"
        )
    )
    r = Ref(p)
    GC.@preserve r op begin
        dst = Ptr{T}(pointer(op.bytes))
        src = Base.unsafe_convert(Ptr{T}, r)
        unsafe_store!(dst, unsafe_load(src))
    end
    return op
end

Base.length(op::OpaqueParams) = length(op.bytes)

function Base.show(io::IO, op::OpaqueParams)
    return print(
        io, "OpaqueParams(", length(op.bytes), " bytes, typeid=0x",
        string(op.typeid; base = 16), ")"
    )
end

# ---------------------------------------------------------------------------
# OpaqueRef — companion container for non-isbits payloads.
# ---------------------------------------------------------------------------

"""
    OpaqueRef

Type-stable container for an arbitrary payload (need not be `isbits`). Like
[`OpaqueParams`](@ref), the wrapper type is fixed regardless of the payload, so
the solver sees a uniform `typeof(p)`. The payload is recovered inside the user
callback via [`unpack`](@ref) with a `::T` assertion.

Backed by `Base.RefValue{Any}`. The payload is held by reference: mutating the
underlying object after `pack_any` is observable to the unpacker.

Fields:
- `ref::Base.RefValue{Any}` — boxed slot for the payload.
- `typeid::UInt`             — `objectid` of the concrete type used at pack time, for optional checking.

Cost notes:
- Unpack of a mutable / non-`isbits` payload (Vector, Dict, mutable struct…)
  is a pointer load + a type check, no allocation.
- Unpack of an `isbits` payload through `OpaqueRef` involves a box/unbox round
  trip and *can* allocate; prefer `OpaqueParams` when the payload is `isbits`.
"""
struct OpaqueRef
    ref::Base.RefValue{Any}
    typeid::UInt
end

"""
    pack_any(x) -> OpaqueRef

Wrap `x` (of any type) in a fresh `OpaqueRef`. The payload is held by reference;
no copy is made.
"""
function pack_any(x::T) where {T}
    return OpaqueRef(Ref{Any}(x), objectid(T))
end

# Convenience constructor.
OpaqueRef(x) = pack_any(x)

"""
    unpack(op::OpaqueRef, ::Type{T}) -> T

Read the payload back as a `T`. Type-stable because of the trailing `::T`
assertion. For non-`isbits` payloads this is a pointer load + type-tag check
with no allocation.
"""
@inline function unpack(op::OpaqueRef, ::Type{T}) where {T}
    return op.ref[]::T
end

"""
    unpack_checked(op::OpaqueRef, ::Type{T}) -> T

Like `unpack` but also verifies `T` matches what was packed via `objectid`.
"""
@inline function unpack_checked(op::OpaqueRef, ::Type{T}) where {T}
    objectid(T) == op.typeid || throw(
        ArgumentError(
            "typeid mismatch: OpaqueRef was packed with a different concrete type"
        )
    )
    return unpack(op, T)
end

"""
    repack!(op::OpaqueRef, x) -> op

Replace the payload in place. Requires the new payload to have the same concrete
type as the original (so the wrapper's `typeid` remains valid). Use `pack_any`
to make a new container with a different payload type.
"""
function repack!(op::OpaqueRef, x::T) where {T}
    objectid(T) == op.typeid || throw(
        ArgumentError(
            "repack! on OpaqueRef requires the same concrete type as pack_any " *
                "(got $T). Use pack_any to make a new container."
        )
    )
    op.ref[] = x
    return op
end

function Base.show(io::IO, op::OpaqueRef)
    return print(io, "OpaqueRef(typeid=0x", string(op.typeid; base = 16), ")")
end

# ---------------------------------------------------------------------------
# OpaqueVoid — callable wrapper that recovers the concrete type at the f-boundary
# ---------------------------------------------------------------------------

"""
    OpaqueVoid{P, F}

Callable wrapper that recovers a concrete parameter type `P` at the boundary of
a user callback. `OpaqueVoid` holds a function `f` and, when invoked with an
[`OpaqueParams`](@ref) in the parameter slot, unpacks it back to a `P` value
with [`unsafe_unpack`](@ref) before forwarding to `f`. It returns `nothing`,
mirroring the in-place SciML callback convention (the result is written into
the first argument).

The purpose is to keep a callable's *type signature* uniform on `OpaqueParams`
regardless of the underlying payload type `P`, so that a single
compiled/precompiled code path — e.g. a solver's function-wrapped RHS — is
shared across problems whose parameter struct types differ. The unpack is a
single `unsafe_load` and is type-stable with no allocation, so the wrapped `f`
still runs fully specialized on `P`.

Two SciML in-place shapes are supported, both with the parameter in the third
positional slot:

  - `f(a, u, p, t)` — e.g. an ODE RHS `rhs!(du, u, p, t)`, Jacobian
    `jac!(J, u, p, t)`, or time gradient `tgrad!(dT, u, p, t)`.
  - `f(a, u, p)` — e.g. an in-place nonlinear residual `res!(du, u, p)`.

`P` must match the concrete type that was `pack`ed into the `OpaqueParams`;
[`unsafe_unpack`](@ref) does not check. Construct with `OpaqueVoid(P, f)`.

```jldoctest
julia> using RespecializeParams

julia> nt = (k = 2.0,);

julia> op = pack(nt);

julia> w = OpaqueVoid(typeof(nt), (du, u, p, t) -> (du[1] = -p.k * u[1]; nothing));

julia> du = [0.0]; w(du, [1.0], op, 0.0); du
1-element Vector{Float64}:
 -2.0
```
"""
struct OpaqueVoid{P, F}
    f::F
end

OpaqueVoid(::Type{P}, f::F) where {P, F} = OpaqueVoid{P, F}(f)

# 4-arg SciML shape: f(a, u, p, t) with p in slot 3.
@inline function (v::OpaqueVoid{P})(a, u, op::OpaqueParams, t) where {P}
    p = unsafe_unpack(op, P)
    v.f(a, u, p, t)
    return nothing
end

# 3-arg SciML shape: f(a, u, p) with p in slot 3.
@inline function (v::OpaqueVoid{P})(a, u, op::OpaqueParams) where {P}
    p = unsafe_unpack(op, P)
    v.f(a, u, p)
    return nothing
end

function Base.show(io::IO, ::OpaqueVoid{P, F}) where {P, F}
    return print(io, "OpaqueVoid{", P, "}(", F, ")")
end

end # module
