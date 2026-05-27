using RespecializeParams
using Test

# ---------------------------------------------------------------------------
# Roundtrip on assorted non-isbits payloads
# ---------------------------------------------------------------------------

struct VecP                         # struct holding a Vector → not isbits
    coefs::Vector{Float64}
    scale::Float64
end

mutable struct MutP                 # mutable struct → not isbits
    a::Float64
    b::Int
end

@testset "OpaqueRef roundtrip on non-isbits payloads" begin
    payloads = (
        [1.0, 2.0, 3.0],
        Dict(:a => 1, :b => 2),
        "hello",
        VecP([1.0, 2.0, 3.0], 0.5),
        MutP(1.5, 7),
        (x -> 2x),                  # closures are non-isbits
    )
    for p in payloads
        op = pack_any(p)
        @test op isa OpaqueRef
        # The unpacked value is the *same* object (===) for mutable/heap types.
        @test unpack(op, typeof(p)) === p
        @test unpack_checked(op, typeof(p)) === p
    end
end

# ---------------------------------------------------------------------------
# Identity preservation: mutating the payload propagates through the ref.
# ---------------------------------------------------------------------------
@testset "OpaqueRef preserves identity (mutation propagates)" begin
    v = [1.0, 2.0, 3.0]
    op = pack_any(v)
    @test unpack(op, Vector{Float64}) === v
    v[1] = 99.0
    @test unpack(op, Vector{Float64})[1] == 99.0

    m = MutP(1.5, 7)
    op2 = pack_any(m)
    m.a = 42.0
    @test unpack(op2, MutP).a == 42.0
end

# ---------------------------------------------------------------------------
# Uniform wrapper type across totally different payloads.
# ---------------------------------------------------------------------------
@testset "OpaqueRef uniform wrapper type" begin
    op1 = pack_any([1.0, 2.0])
    op2 = pack_any(Dict(:k => 1))
    op3 = pack_any(VecP([1.0], 1.0))
    @test typeof(op1) === typeof(op2) === typeof(op3) === OpaqueRef
    @test op1.typeid != op2.typeid != op3.typeid
end

# ---------------------------------------------------------------------------
# Type stability + zero allocations in a realistic callback.
# ---------------------------------------------------------------------------
function rhs_vec!(du, u, op::OpaqueRef, t)
    p = unpack(op, VecP)
    @inbounds du[1] = -p.scale * (p.coefs[1] * u[1] + p.coefs[2] * u[2])
    @inbounds du[2] = -p.scale * (p.coefs[3] * u[1] + p.coefs[1] * u[2])
    return nothing
end

@testset "OpaqueRef: type-stable and alloc-free in a callback" begin
    op = pack_any(VecP([1.0, 0.5, 0.25], 0.1))
    u = [1.0, 0.5]
    du = zero(u)
    rhs_vec!(du, u, op, 0.0)        # warm
    @inferred unpack(op, VecP)
    @test 0 == @allocated rhs_vec!(du, u, op, 0.0)
end

# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
@testset "OpaqueRef error paths" begin
    op = pack_any([1.0, 2.0])
    @test_throws TypeError    unpack(op, Vector{Int})           # type assertion
    @test_throws ArgumentError unpack_checked(op, Vector{Int})  # typeid mismatch

    # repack! requires same concrete type
    @test_throws ArgumentError repack!(op, [1, 2])              # Vector{Int}
    @test_throws ArgumentError repack!(op, "string")
    # same type works
    repack!(op, [9.0, 8.0, 7.0])
    @test unpack(op, Vector{Float64}) == [9.0, 8.0, 7.0]
end

# ---------------------------------------------------------------------------
# pack on a non-isbits value via the isbits-only API errors clearly.
# ---------------------------------------------------------------------------
@testset "pack on non-isbits errors; pack_any handles it" begin
    @test_throws ArgumentError pack([1.0, 2.0])
    @test_throws ArgumentError pack(VecP([1.0], 1.0))

    # pack_any is the right tool here.
    op = pack_any(VecP([1.0, 2.0], 0.5))
    @test op isa OpaqueRef
end
