using RespecializeParams
using NonlinearSolve
using SciMLBase
using Test

# Two different parameter struct types for the same nonlinear-system signature.
struct QuadP
    a::Float64
    b::Float64
    c::Float64
end

struct LineP
    m::Float64
    k::Float64
end

# Callable struct: typeof(rhs) carries the concrete payload type, typeof(p) stays OpaqueParams.
struct OpaqueResid!{T, F} <: Function
    f!::F
end
OpaqueResid!(::Type{T}, f!::F) where {T, F} = OpaqueResid!{T, F}(f!)

@inline function (r::OpaqueResid!{T})(du, u, op::OpaqueParams) where {T}
    p = unpack(op, T)
    r.f!(du, u, p)
    return nothing
end

# In-place residual kernels.
function quad_kernel!(du, u, p::QuadP)
    @inbounds du[1] = p.a * u[1]^2 + p.b * u[1] + p.c
    return nothing
end

function line_kernel!(du, u, p::LineP)
    @inbounds du[1] = p.m * u[1] + p.k
    return nothing
end

# Non-opaque baseline.
function quad_plain!(du, u, p::QuadP)
    @inbounds du[1] = p.a * u[1]^2 + p.b * u[1] + p.c
    return nothing
end

# ---------------------------------------------------------------------------
@testset "NonlinearSolve: numerical agreement with baseline" begin
    p = QuadP(1.0, -3.0, 2.0)               # roots at x = 1, 2
    u0 = [3.0]                               # converges to root x = 2

    prob_plain = NonlinearProblem(quad_plain!, u0, p)
    prob_opaque = NonlinearProblem(OpaqueResid!(QuadP, quad_kernel!), u0, pack(p))

    sol_plain = solve(prob_plain, NewtonRaphson(), abstol = 1.0e-12)
    sol_opaque = solve(prob_opaque, NewtonRaphson(), abstol = 1.0e-12)

    @test sol_opaque.u[1] ≈ sol_plain.u[1] atol = 1.0e-10
    @test sol_opaque.u[1] ≈ 2.0 atol = 1.0e-10
end

# ---------------------------------------------------------------------------
@testset "NonlinearSolve: prob.p uniform across payloads" begin
    prob_q = NonlinearProblem(
        OpaqueResid!(QuadP, quad_kernel!),
        [3.0], pack(QuadP(1.0, -3.0, 2.0))
    )
    prob_l = NonlinearProblem(
        OpaqueResid!(LineP, line_kernel!),
        [0.0], pack(LineP(2.0, -4.0))
    )

    @test typeof(prob_q.p) === typeof(prob_l.p) === OpaqueParams

    sol_l = solve(prob_l, NewtonRaphson(), abstol = 1.0e-12)
    @test sol_l.u[1] ≈ 2.0 atol = 1.0e-10     # m*x + k = 0 => x = -k/m = 2
end

# ---------------------------------------------------------------------------
@testset "NonlinearSolve: residual call is non-allocating (after unpack)" begin
    rhs = OpaqueResid!(QuadP, quad_kernel!)
    op = pack(QuadP(1.0, -3.0, 2.0))
    u = [3.0]
    du = zero(u)
    rhs(du, u, op)  # warm
    @test 0 == @allocated rhs(du, u, op)
end
