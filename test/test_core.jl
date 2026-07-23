using RespecializeParams
using Test

# ---------------------------------------------------------------------------
# Roundtrip on assorted isbits payloads
# ---------------------------------------------------------------------------

struct PendulumP
    g::Float64
    L::Float64
    m::Float64
end

struct LotkaP
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
end

struct MixedP
    n::Int32
    flag::Bool
    x::Float64
    y::Float32
end

@testset "roundtrip on isbits payloads" begin
    for p in (
            1.0,
            (1.0, 2.0, 3.0),
            (a = 1.0, b = 2, c = 3.5f0),
            PendulumP(9.81, 1.0, 0.5),
            LotkaP(1.5, 1.0, 3.0, 1.0),
            MixedP(Int32(7), true, 3.14, 2.5f0),
            ntuple(i -> Float64(i), Val(8)),
        )
        op = pack(p)
        T = typeof(p)
        @test op isa OpaqueParams
        @test length(op) == sizeof(T)
        @test unpack(op, T) === p
        @test unsafe_unpack(op, T) === p
        @test unpack_checked(op, T) === p
    end
end

# ---------------------------------------------------------------------------
# Uniform wrapper type — different payload types pack to the SAME container type
# ---------------------------------------------------------------------------

@testset "uniform wrapper type" begin
    op1 = pack(PendulumP(9.81, 1.0, 0.5))
    op2 = pack(LotkaP(1.5, 1.0, 3.0, 1.0))
    op3 = pack((a = 1.0, b = 2))

    @test typeof(op1) === typeof(op2) === typeof(op3) === OpaqueParams
    # but the underlying byte lengths and typeids differ
    @test length(op1) != length(op2)
    @test op1.typeid != op2.typeid != op3.typeid
end

# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

mutable struct NotBits
    x::Vector{Float64}
end

@testset "error paths" begin
    @test_throws ArgumentError pack(NotBits([1.0, 2.0]))

    op = pack(PendulumP(9.81, 1.0, 0.5))
    # size mismatch
    @test_throws ArgumentError unpack(op, LotkaP)
    # typeid mismatch
    bad = pack((a = 1.0, b = 2.0, c = 3.0))   # same byte count as PendulumP (24)
    @test length(bad) == sizeof(PendulumP)
    @test_throws ArgumentError unpack_checked(bad, PendulumP)
end

# ---------------------------------------------------------------------------
# Type stability and zero-allocation of unpack inside a real callback
# ---------------------------------------------------------------------------

# A user f that pretends it's a SciML rhs.
function rhs!(du, u, op::OpaqueParams, t)
    p = unpack(op, PendulumP)
    @inbounds du[1] = u[2]
    @inbounds du[2] = -(p.g / p.L) * sin(u[1])
    return nothing
end

function rhs_unsafe!(du, u, op::OpaqueParams, t)
    p = unsafe_unpack(op, PendulumP)
    @inbounds du[1] = u[2]
    @inbounds du[2] = -(p.g / p.L) * sin(u[1])
    return nothing
end

@testset "type stability and allocation-freedom in a callback" begin
    op = pack(PendulumP(9.81, 1.0, 0.5))
    u = [0.1, 0.0]
    du = zero(u)

    # warm up
    rhs!(du, u, op, 0.0)
    rhs_unsafe!(du, u, op, 0.0)

    # type stability
    @inferred unpack(op, PendulumP)
    @inferred unsafe_unpack(op, PendulumP)

    # allocations
    @test 0 == @allocated rhs!(du, u, op, 0.0)
    @test 0 == @allocated rhs_unsafe!(du, u, op, 0.0)
end

# ---------------------------------------------------------------------------
# repack! mutates in place without changing the container's identity / size
# ---------------------------------------------------------------------------

@testset "repack!" begin
    op = pack(PendulumP(9.81, 1.0, 0.5))
    bytes_id = objectid(op.bytes)
    repack!(op, PendulumP(3.7, 2.0, 1.0))
    @test objectid(op.bytes) == bytes_id          # same backing array
    @test unpack(op, PendulumP) === PendulumP(3.7, 2.0, 1.0)

    # repack with a different-size type errors (LotkaP is 4×Float64 = 32 bytes, PendulumP is 24)
    @test sizeof(LotkaP) != sizeof(PendulumP)
    @test_throws ArgumentError repack!(op, LotkaP(1.0, 2.0, 3.0, 4.0))
end

# ---------------------------------------------------------------------------
# OpaqueVoid — recovers the concrete type at the callback boundary
# ---------------------------------------------------------------------------

pendulum_rhs!(du, u, p::PendulumP, t) = (
    @inbounds du[1] = u[2];
    @inbounds du[2] = -(p.g / p.L) * sin(u[1]); nothing
)

# in-place nonlinear residual shape: res!(du, u, p)
lotka_res!(res, u, p::LotkaP) = (@inbounds res[1] = p.α * u[1] - p.β; nothing)

# A non-isbits parameter type (has a Vector field), routed through OpaqueRef.
struct VecP
    ks::Vector{Float64}
end
vecp_rhs!(du, u, p::VecP, t) = (@inbounds du[1] = -p.ks[1] * u[1]; nothing)

@testset "OpaqueVoid forwards after unpacking" begin
    p = PendulumP(9.81, 1.0, 0.5)
    op = pack(p)
    w = OpaqueVoid(PendulumP, pendulum_rhs!)
    @test w isa OpaqueVoid{PendulumP}

    du = zeros(2)
    u = [0.3, 0.0]
    w(du, u, op, 0.0)
    ref = zeros(2)
    pendulum_rhs!(ref, u, p, 0.0)
    @test du == ref

    # 3-arg (nonlinear residual) shape
    q = LotkaP(1.5, 0.5, 0.0, 0.0)
    opq = pack(q)
    wq = OpaqueVoid(LotkaP, lotka_res!)
    res = zeros(1)
    wq(res, [2.0], opq)
    @test res[1] ≈ q.α * 2.0 - q.β
end

@testset "OpaqueVoid through OpaqueRef (non-isbits payload)" begin
    p = VecP([2.0])
    op = pack_any(p)                      # non-isbits → OpaqueRef
    @test op isa OpaqueRef
    w = OpaqueVoid(VecP, vecp_rhs!)

    du = [0.0]
    w(du, [3.0], op, 0.0)                  # 4-arg shape on OpaqueRef
    @test du[1] ≈ -6.0

    # 3-arg shape on OpaqueRef
    res_fn!(res, u, p::VecP) = (@inbounds res[1] = p.ks[1] - u[1]; nothing)
    wr = OpaqueVoid(VecP, res_fn!)
    res = [0.0]
    wr(res, [0.5], op)
    @test res[1] ≈ 1.5

    # OpaqueRef preserves identity, so payload mutation is visible through the wrapper
    p.ks[1] = 10.0
    w(du, [1.0], op, 0.0)
    @test du[1] ≈ -10.0
end

@testset "OpaqueVoid type is uniform across payload types" begin
    # Different concrete P give different OpaqueVoid types (P is a type param),
    # but the *argument* type at the call site is always OpaqueParams — that is
    # the property callers rely on to share one compiled path.
    w1 = OpaqueVoid(PendulumP, pendulum_rhs!)
    @test typeof(pack(PendulumP(1.0, 1.0, 1.0))) ===
        typeof(pack(LotkaP(1.0, 2.0, 3.0, 4.0))) === OpaqueParams
    # method dispatches on OpaqueParams regardless of the packed payload
    m = only(methods(w1, Tuple{Any, Any, OpaqueParams, Any}))
    @test m.sig <: Tuple{OpaqueVoid, Any, Any, OpaqueParams, Any}
end

@testset "OpaqueVoid is allocation-free" begin
    op = pack(PendulumP(9.81, 1.0, 0.5))
    w = OpaqueVoid(PendulumP, pendulum_rhs!)
    du = zeros(2)
    u = [0.3, 0.0]
    # measure behind a function barrier so @allocated does not capture
    # non-const global access
    meas(w, du, u, op, t) = (w(du, u, op, t); @allocated w(du, u, op, t))
    @test meas(w, du, u, op, 0.0) == 0
end

# ---------------------------------------------------------------------------
# Solver-integration helpers: container selection, packing, signature surgery
# ---------------------------------------------------------------------------

@testset "opaque_container_type / pack_auto" begin
    @test opaque_container_type(PendulumP) === OpaqueParams   # isbits
    @test opaque_container_type(VecP) === OpaqueRef           # non-isbits
    @test opaque_container_type(NamedTuple{(:k,), Tuple{Float64}}) === OpaqueParams

    @test pack_auto(PendulumP(9.81, 1.0, 0.5)) isa OpaqueParams
    @test pack_auto(VecP([1.0])) isa OpaqueRef
    # roundtrip through the auto-selected container
    @test unpack(pack_auto(PendulumP(9.81, 1.0, 0.5)), PendulumP) ===
        PendulumP(9.81, 1.0, 0.5)
    @test unpack(pack_auto(VecP([2.0])), VecP).ks == [2.0]
end

@testset "opaque_signature substitutes the p slot" begin
    # 4-arg (out, u, p, t)
    @test opaque_signature(
        Tuple{Vector{Float64}, Vector{Float64}, PendulumP, Float64}, OpaqueParams
    ) === Tuple{Vector{Float64}, Vector{Float64}, OpaqueParams, Float64}
    # 3-arg (out, u, p)
    @test opaque_signature(Tuple{Vector{Float64}, Vector{Float64}, LotkaP}, OpaqueParams) ===
        Tuple{Vector{Float64}, Vector{Float64}, OpaqueParams}
    # OpaqueRef container
    @test opaque_signature(Tuple{Vector{Float64}, Vector{Float64}, VecP, Float64}, OpaqueRef) ===
        Tuple{Vector{Float64}, Vector{Float64}, OpaqueRef, Float64}
end

# `wrap_void_opaque` lives in the FunctionWrappersWrappers extension.
using FunctionWrappersWrappers

@testset "wrap_void_opaque installs OpaqueVoid behind a FunctionWrapper" begin
    p = PendulumP(9.81, 1.0, 0.5)
    natural_sig = Tuple{Vector{Float64}, Vector{Float64}, PendulumP, Float64}
    w = wrap_void_opaque(pendulum_rhs!, PendulumP, (natural_sig,))
    @test w isa FunctionWrappersWrappers.FunctionWrappersWrapper

    # dispatches on the de-specialized (OpaqueParams in slot 3) signature
    op = pack(p)
    du = zeros(2)
    u = [0.3, 0.0]
    w(du, u, op, 0.0)
    ref = zeros(2)
    pendulum_rhs!(ref, u, p, 0.0)
    @test du == ref

    # two different isbits payload types produce the SAME wrapper type,
    # which is the whole point (one compiled path shared across parameter types)
    w2 = wrap_void_opaque(
        (du, u, q::LotkaP, t) -> (du[1] = q.α; nothing), LotkaP,
        (Tuple{Vector{Float64}, Vector{Float64}, LotkaP, Float64},),
    )
    @test typeof(w) === typeof(w2)

    # 3-arg residual shape
    wr = wrap_void_opaque(lotka_res!, LotkaP, (Tuple{Vector{Float64}, Vector{Float64}, LotkaP},))
    res = zeros(1)
    q = LotkaP(1.5, 0.5, 0.0, 0.0)
    wr(res, [2.0], pack(q))
    @test res[1] ≈ q.α * 2.0 - q.β

    # non-isbits payload → wrapper signature carries OpaqueRef, dispatches on pack_any
    wv = wrap_void_opaque(vecp_rhs!, VecP, (Tuple{Vector{Float64}, Vector{Float64}, VecP, Float64},))
    du2 = [0.0]
    wv(du2, [3.0], pack_any(VecP([2.0])), 0.0)
    @test du2[1] ≈ -6.0
end
