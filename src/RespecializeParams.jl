module RespecializeParams

import FunctionWrappersWrappers

export OpaqueParams, OpaqueRef, OpaqueVoid,
    pack, pack_any, pack_auto, unpack, unsafe_unpack, unpack_checked, repack!, payload,
    opaque_container_type, opaque_signature, wrap_void_opaque

"""
    OpaqueParams

Type-stable container for an `isbits` value. The wrapper type is fixed regardless
of the underlying payload type, so callers (e.g. ODE/Nonlinear solvers) that
dispatch on `typeof(p)` always hit the same precompiled code path. The payload
type is recovered inside the user-supplied callback via [`unpack`](@ref), which
reinterprets the bytes back to the concrete type with no allocation.

Fields:
- `bytes::Vector{UInt8}`: Raw payload bytes. The vector is heap allocated and its
  length is fixed for each instance.
- `typeid::UInt`: `objectid` of the concrete type used at [`pack`](@ref) time. It
  is used by [`unpack_checked`](@ref), not by [`unpack`](@ref).

`OpaqueParams` is intentionally **not** parametric: that's the whole point. If you
need a parametric wrapper (e.g. for dispatch in your own callback hierarchy) just
wrap it: `struct MyParams{T}; op::OpaqueParams; end`.

# Examples

```jldoctest
julia> using RespecializeParams

julia> p = (rate = 2.0, count = 3);

julia> typeof(pack(p))
OpaqueParams
```
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

# Arguments
- `p`: An `isbits` value to copy into opaque storage.

# Throws
- `ArgumentError`: `typeof(p)` is not an `isbitstype`.

# Examples

```jldoctest
julia> using RespecializeParams

julia> unpack(pack((gain = 2.0,)), NamedTuple{(:gain,), Tuple{Float64}})
(gain = 2.0,)
```
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

# Arguments
- `op`: An [`OpaqueParams`](@ref) created by [`pack`](@ref).
- `T`: The requested `isbits` payload type. Its size must equal `length(op.bytes)`.

# Throws
- `ArgumentError`: `T` is not an `isbitstype` or has a different size from the
  stored payload.
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

# Arguments
- `op`: An [`OpaqueParams`](@ref) created by [`pack`](@ref).
- `T`: The `isbits` type used to interpret the stored bytes.

# Safety
`T` must have exactly the size and layout used when `op` was packed. Prefer
[`unpack`](@ref) or [`unpack_checked`](@ref) unless that invariant is established
by the caller.
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

# Arguments
- `op`: An [`OpaqueParams`](@ref) created by [`pack`](@ref).
- `T`: The expected concrete payload type.

# Throws
- `ArgumentError`: `T` differs from the type originally supplied to [`pack`](@ref),
  or its storage size differs from the stored payload.
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

# Arguments
- `op`: An [`OpaqueParams`](@ref) whose backing storage is updated in place.
- `p`: A new `isbits` value with the same storage size as the existing payload.

# Throws
- `ArgumentError`: `p` is not `isbits` or its size differs from the stored payload.
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

The payload is held in a boxed field. Mutating a mutable object after
[`pack_any`](@ref) is observable to the unpacker.

Fields:
- `value::Any`: Boxed payload slot.
- `typeid::UInt`: `objectid` of the concrete type used at pack time, for optional
  checking.

Cost notes:
- Unpack of a mutable / non-`isbits` payload (Vector, Dict, mutable struct…)
  is a pointer load + a type check, no allocation.
- Unpack of an `isbits` payload through `OpaqueRef` involves a box/unbox round
  trip and *can* allocate; prefer `OpaqueParams` when the payload is `isbits`.

# Examples

```jldoctest
julia> using RespecializeParams

julia> op = pack_any([1.0, 2.0]);

julia> unpack(op, Vector{Float64})
2-element Vector{Float64}:
 1.0
 2.0
```
"""
mutable struct OpaqueRef
    value::Any
    typeid::UInt
end

"""
    pack_any(x) -> OpaqueRef

Wrap `x` (of any type) in a fresh `OpaqueRef`. The payload is held by reference;
no copy is made.

# Arguments
- `x`: The payload to store. It may have any concrete type.
"""
function pack_any(x::T) where {T}
    return OpaqueRef(x, objectid(T))
end

# Convenience constructor.
"""
    OpaqueRef(x)

Equivalent to [`pack_any`](@ref). This convenience constructor stores `x` without
copying it.

# Arguments
- `x`: Any payload value.
"""
OpaqueRef(x) = pack_any(x)

"""
    unpack(op::OpaqueRef, ::Type{T}) -> T

Read the payload back as a `T`. Type-stable because of the trailing `::T`
assertion. For non-`isbits` payloads this is a pointer load + type-tag check
with no allocation.

# Arguments
- `op`: An [`OpaqueRef`](@ref) created by [`pack_any`](@ref).
- `T`: The expected concrete payload type.

# Throws
- `TypeError`: The stored value is not a `T`.
"""
@inline function unpack(op::OpaqueRef, ::Type{T}) where {T}
    return op.value::T
end

"""
    unpack_checked(op::OpaqueRef, ::Type{T}) -> T

Like `unpack` but also verifies `T` matches what was packed via `objectid`.

# Arguments
- `op`: An [`OpaqueRef`](@ref) created by [`pack_any`](@ref).
- `T`: The expected concrete payload type.

# Throws
- `ArgumentError`: `T` differs from the type originally supplied to
  [`pack_any`](@ref).
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
    payload(op::OpaqueRef) -> Any

Read the payload back without naming its type. Returns `Any`, so this is
deliberately **not** type-stable — it exists for cold paths that must inspect or
transform the payload while having no concrete type available (parameter
promotion, initialization, symbolic indexing). Prefer [`unpack`](@ref) with a
concrete `T` anywhere the type is known, and always on a hot path: `unpack` is a
pointer load with a `::T` assertion and stays inference-friendly, whereas
`payload` forces the caller to handle an `Any`.

# Arguments
- `op`: An [`OpaqueRef`](@ref) whose payload is read.

```jldoctest
julia> using RespecializeParams

julia> op = pack_any([1.0, 2.0]);

julia> payload(op)
2-element Vector{Float64}:
 1.0
 2.0
```
"""
@inline payload(op::OpaqueRef) = op.value

"""
    repack!(op::OpaqueRef, x) -> op

Replace the payload in place. Requires the new payload to have the same concrete
type as the original (so the wrapper's `typeid` remains valid). Use `pack_any`
to make a new container with a different payload type.

# Arguments
- `op`: An [`OpaqueRef`](@ref) whose payload slot is replaced.
- `x`: A replacement with the same concrete type as the original payload.

# Throws
- `ArgumentError`: `x` has a different concrete type from the original payload.
"""
function repack!(op::OpaqueRef, x::T) where {T}
    objectid(T) == op.typeid || throw(
        ArgumentError(
            "repack! on OpaqueRef requires the same concrete type as pack_any " *
                "(got $T). Use pack_any to make a new container."
        )
    )
    op.value = x
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
[`OpaqueParams`](@ref) or [`OpaqueRef`](@ref) in the parameter slot, unpacks it
back to a `P` value before forwarding to `f`. It returns `nothing`, mirroring
the in-place SciML callback convention (the result is written into the first
argument).

The purpose is to keep a callable's *type signature* uniform on the container
type regardless of the underlying payload type `P`, so that a single
compiled/precompiled code path — e.g. a solver's function-wrapped RHS — is
shared across problems whose parameter struct types differ. Both containers
give a fixed wrapper type: `OpaqueParams` unifies all `isbits` payloads
(unpacked with [`unsafe_unpack`](@ref), a single `unsafe_load`) and `OpaqueRef`
unifies all payloads (unpacked with [`unpack`](@ref), a pointer load + type-tag
assertion). Either way the unpack is type-stable and allocation-free for the
container's intended payloads, so the wrapped `f` still runs fully specialized
on `P`.

Two SciML in-place shapes are supported, both with the parameter in the third
positional slot:

  - `f(a, u, p, t)` — e.g. an ODE RHS `rhs!(du, u, p, t)`, Jacobian
    `jac!(J, u, p, t)`, or time gradient `tgrad!(dT, u, p, t)`.
  - `f(a, u, p)` — e.g. an in-place nonlinear residual `res!(du, u, p)`.

`P` must match the concrete type that was `pack`ed into the `OpaqueParams`;
[`unsafe_unpack`](@ref) does not check. Construct with `OpaqueVoid(P, f)`.

# Fields
- `f`: The in-place callback receiving the recovered parameter value.

# Constructor arguments
- `P`: The concrete type recovered from the opaque parameter container.
- `f`: A callable supporting `f(out, u, p, t)`, `f(out, u, p)`, or both.

# Interface contract
`OpaqueVoid` is a developer-facing callable interface for SciML solver
integrations. Generic callers may invoke only the documented 3- and 4-argument
in-place shapes above; implementations must mutate `out`, return `nothing`, and
keep the opaque parameter in positional slot three. `OpaqueParams` callers must
use the same `P` supplied at construction; `OpaqueRef` callers require a payload
whose concrete type is `P`.

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

"""
    OpaqueVoid(P, f)

Construct an [`OpaqueVoid`](@ref) that recovers a parameter of concrete type `P`
before invoking `f` through the documented in-place callback interface.

# Arguments
- `P`: The concrete payload type.
- `f`: A callback supporting the 3- or 4-argument in-place shape described by
  [`OpaqueVoid`](@ref).
"""
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

# OpaqueRef variants — for non-isbits payloads. `unpack(::OpaqueRef, P)` is the
# minimal type-stable read (pointer load + `::P` assertion; no `unsafe_unpack`
# exists or is needed for OpaqueRef).
@inline function (v::OpaqueVoid{P})(a, u, op::OpaqueRef, t) where {P}
    p = unpack(op, P)
    v.f(a, u, p, t)
    return nothing
end

@inline function (v::OpaqueVoid{P})(a, u, op::OpaqueRef) where {P}
    p = unpack(op, P)
    v.f(a, u, p)
    return nothing
end

function Base.show(io::IO, ::OpaqueVoid{P, F}) where {P, F}
    return print(io, "OpaqueVoid{", P, "}(", F, ")")
end

# ---------------------------------------------------------------------------
# Solver-integration helpers: choose the container, pack, and build the
# de-specialized callback signature. These are the shared pieces a SciML solver
# stack needs to install `OpaqueVoid` under the `AutoDePSpecialize`
# specialization level (used by DiffEqBase and NonlinearSolve).
# ---------------------------------------------------------------------------

"""
    opaque_container_type(::Type{P}) -> Type

The fixed opaque container type that erases a parameter of concrete type `P`:
[`OpaqueParams`](@ref) when `P` is an `isbitstype` (packed via [`pack`](@ref)),
otherwise [`OpaqueRef`](@ref) (packed via [`pack_any`](@ref)). This is the type
that appears in the parameter slot of a de-specialized callback signature.

# Arguments
- `P`: A concrete payload type.

# Developer API
This helper is for solver integrations that construct de-specialized callback
signatures. End-user code should normally call [`pack_auto`](@ref) instead.
"""
@inline opaque_container_type(::Type{P}) where {P} =
    isbitstype(P) ? OpaqueParams : OpaqueRef

"""
    pack_auto(p) -> OpaqueParams or OpaqueRef

Pack `p` into whichever container [`opaque_container_type`](@ref) selects for its
type: [`pack`](@ref) for `isbits` payloads, [`pack_any`](@ref) otherwise. A
uniform "erase this parameter" entry point regardless of the payload's bits-ness.

# Arguments
- `p`: Any payload value to store in the appropriate opaque container.
"""
@inline pack_auto(p) = isbits(p) ? pack(p) : pack_any(p)

"""
    opaque_signature(::Type{S}, ::Type{C}) -> Type

Return the SciML in-place callback signature `S` (a `Tuple` type) with its
parameter slot — the third positional argument — replaced by the opaque
container type `C`. Supports the 4-argument `f(out, u, p, t)` and 3-argument
`f(out, u, p)` shapes handled by [`OpaqueVoid`](@ref).

# Arguments
- `S`: A 3- or 4-element `Tuple` type whose third element is the natural parameter
  type.
- `C`: The opaque container type to place in the third element.

# Throws
- `MethodError`: `S` is not one of the supported 3- or 4-argument in-place callback
  signatures.

# Developer API
This helper is for solver integrations that construct callback signatures. It
does not validate callback behavior; callers must uphold the [`OpaqueVoid`](@ref)
interface contract.
"""
@inline opaque_signature(::Type{Tuple{A, B, P, T}}, ::Type{C}) where {A, B, P, T, C} =
    Tuple{A, B, C, T}
@inline opaque_signature(::Type{Tuple{A, B, P}}, ::Type{C}) where {A, B, P, C} =
    Tuple{A, B, C}

"""
    wrap_void_opaque(ff, ::Type{P}, sigs::Tuple) -> FunctionWrappersWrapper

Install `OpaqueVoid(P, ff)` behind a `FunctionWrappersWrapper` whose signature(s)
have their parameter slot de-specialized to [`opaque_container_type`](@ref)`(P)`.
`sigs` is a tuple of `Tuple` types — each the *natural* `(out, u, p, t)` or
`(out, u, p)` signature — whose `p` slot is replaced by the opaque container
type via [`opaque_signature`](@ref). The result is a callable that a solver
dispatches on uniformly regardless of `P`, unpacking back to `P` on each call.

This is the shared installer used by SciML solver stacks (DiffEqBase,
NonlinearSolve) under the `AutoDePSpecialize` specialization level.

# Arguments
- `ff`: An in-place callback satisfying the [`OpaqueVoid`](@ref) interface.
- `P`: The concrete payload type recovered before each callback invocation.
- `sigs`: A tuple of natural 3- or 4-argument `Tuple` signature types. Their third
  element is replaced with the result of [`opaque_container_type`](@ref).

# Developer API
This is an extension point for solver packages. Callers must preserve the 3- or
4-argument callback shapes and invoke the resulting wrapper with the container
type selected for `P`; it is not intended as an end-user wrapper API.
"""
function wrap_void_opaque(ff, ::Type{P}, sigs::Tuple) where {P}
    C = opaque_container_type(P)
    opaque_sigs = map(s -> opaque_signature(s, C), sigs)
    nothings = map(_ -> Nothing, sigs)
    return FunctionWrappersWrappers.FunctionWrappersWrapper(
        OpaqueVoid(P, ff), opaque_sigs, nothings,
    )
end

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        payload_value = (rate = 2.0, count = 3)
        packed = pack(payload_value)
        unpack(packed, typeof(payload_value))
        unpack_checked(packed, typeof(payload_value))
        opaque_container_type(typeof(payload_value))
        pack_auto(payload_value)
        opaque_signature(
            Tuple{Vector{Float64}, Vector{Float64}, typeof(payload_value), Float64},
            OpaqueParams,
        )
        ref_packed = pack_any([1, 2])
        unpack(ref_packed, Vector{Int})
        pack_auto([1, 2])
    end
end

end # module
