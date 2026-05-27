using RespecializeParams
using OrdinaryDiffEqTsit5
using SciMLBase
using Test

# Two concrete parameter struct types, deliberately different sizes/layouts.
struct LorenzP
    σ::Float64
    ρ::Float64
    β::Float64
end

struct Lorenz96P
    F::Float64
    N::Int          # different field type AND different size from LorenzP
end

# A single rhs taking OpaqueParams. The constant P-type tag is encoded by
# wrapping the rhs in a callable struct parameterized by the concrete type.
# That way `typeof(rhs)` carries the type info (so unpack is a compile-time
# constant) while `typeof(p)` is always OpaqueParams.
struct OpaqueRHS{T,F} <: Function
    f::F
end
OpaqueRHS(::Type{T}, f::F) where {T,F} = OpaqueRHS{T,F}(f)

@inline function (r::OpaqueRHS{T})(du, u, op::OpaqueParams, t) where {T}
    p = unpack(op, T)
    r.f(du, u, p, t)
    return nothing
end

# Plain (non-opaque) versions for the numerical baseline.
function lorenz_plain!(du, u, p::LorenzP, t)
    du[1] = p.σ * (u[2] - u[1])
    du[2] = u[1] * (p.ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - p.β * u[3]
    return nothing
end

# The "inner" kernel that OpaqueRHS will call after unpacking.
function lorenz_kernel!(du, u, p::LorenzP, t)
    du[1] = p.σ * (u[2] - u[1])
    du[2] = u[1] * (p.ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - p.β * u[3]
    return nothing
end

# A second, structurally different kernel.
function l96_kernel!(du, u, p::Lorenz96P, t)
    N = p.N
    @inbounds for i in 1:N
        du[i] = (u[mod1(i + 1, N)] - u[mod1(i - 2, N)]) * u[mod1(i - 1, N)] - u[i] + p.F
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Numerical agreement with the non-opaque baseline.
# ---------------------------------------------------------------------------
@testset "ODE: numerical agreement with baseline" begin
    u0   = [1.0, 0.0, 0.0]
    tspan = (0.0, 5.0)
    p    = LorenzP(10.0, 28.0, 8/3)

    prob_plain   = ODEProblem(lorenz_plain!, u0, tspan, p)
    prob_opaque  = ODEProblem(OpaqueRHS(LorenzP, lorenz_kernel!), u0, tspan, pack(p))

    sol_plain  = solve(prob_plain,  Tsit5(), reltol=1e-9, abstol=1e-9)
    sol_opaque = solve(prob_opaque, Tsit5(), reltol=1e-9, abstol=1e-9)

    @test sol_plain.t  ≈ sol_opaque.t  atol=1e-9
    @test sol_plain.u[end] ≈ sol_opaque.u[end] atol=1e-9
end

# ---------------------------------------------------------------------------
# typeof(prob.p) is fixed across underlying payload types.
# This is the core promise: dispatch sees only OpaqueParams.
# ---------------------------------------------------------------------------
@testset "ODE: prob.p has uniform type across payloads" begin
    u0_lz  = [1.0, 0.0, 0.0]
    u0_96  = collect(range(0.0, 1.0; length = 5))

    prob_lz = ODEProblem(OpaqueRHS(LorenzP,   lorenz_kernel!), u0_lz, (0.0, 1.0),
                          pack(LorenzP(10.0, 28.0, 8/3)))
    prob_96 = ODEProblem(OpaqueRHS(Lorenz96P, l96_kernel!),    u0_96, (0.0, 1.0),
                          pack(Lorenz96P(8.0, 5)))

    @test typeof(prob_lz.p) === typeof(prob_96.p) === OpaqueParams
    @test prob_lz.p isa OpaqueParams
    @test prob_96.p isa OpaqueParams
end

# ---------------------------------------------------------------------------
# Inner rhs is non-allocating after the unpack.
# ---------------------------------------------------------------------------
@testset "ODE: rhs call is non-allocating" begin
    rhs = OpaqueRHS(LorenzP, lorenz_kernel!)
    op  = pack(LorenzP(10.0, 28.0, 8/3))
    u   = [1.0, 0.0, 0.0]
    du  = zero(u)
    rhs(du, u, op, 0.0)  # warm
    @test 0 == @allocated rhs(du, u, op, 0.0)
end

# ---------------------------------------------------------------------------
# In-place repack! lets a user mutate parameters between solves without
# re-allocating or changing dispatch.
# ---------------------------------------------------------------------------
@testset "ODE: repack! between solves" begin
    u0   = [1.0, 0.0, 0.0]
    tspan = (0.0, 1.0)
    op   = pack(LorenzP(10.0, 28.0, 8/3))
    prob = ODEProblem(OpaqueRHS(LorenzP, lorenz_kernel!), u0, tspan, op)

    sol1 = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9)

    repack!(op, LorenzP(8.0, 20.0, 2.0))
    # Same prob object, same p container — mutation is observable.
    sol2 = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9)

    @test sol1.u[end] != sol2.u[end]

    # Baseline comparison with the mutated parameter.
    prob_baseline = ODEProblem(lorenz_plain!, u0, tspan, LorenzP(8.0, 20.0, 2.0))
    sol_baseline  = solve(prob_baseline, Tsit5(), reltol=1e-9, abstol=1e-9)
    @test sol2.u[end] ≈ sol_baseline.u[end] atol=1e-9
end
