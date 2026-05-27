using RespecializeParams
using OrdinaryDiffEqTsit5
using SciMLBase
using Test

# Non-isbits parameter type: a struct containing a Vector.
struct CoeffP
    a::Vector{Float64}    # coefficients
    s::Float64            # scale
end

# Callable struct: T tag goes on the rhs so unpack is a compile-time constant.
struct OpaqueRefRHS{T,F} <: Function
    f::F
end
OpaqueRefRHS(::Type{T}, f::F) where {T,F} = OpaqueRefRHS{T,F}(f)

@inline function (r::OpaqueRefRHS{T})(du, u, op::OpaqueRef, t) where {T}
    p = unpack(op, T)
    r.f(du, u, p, t)
    return nothing
end

# Kernels.
function coeffs_plain!(du, u, p::CoeffP, t)
    @inbounds du[1] = -p.s * (p.a[1] * u[1] + p.a[2] * u[2])
    @inbounds du[2] = -p.s * (p.a[3] * u[1] + p.a[1] * u[2])
    return nothing
end
function coeffs_kernel!(du, u, p::CoeffP, t)
    @inbounds du[1] = -p.s * (p.a[1] * u[1] + p.a[2] * u[2])
    @inbounds du[2] = -p.s * (p.a[3] * u[1] + p.a[1] * u[2])
    return nothing
end

# A second, structurally different non-isbits payload for the uniform-type test.
struct DictP
    d::Dict{Symbol,Float64}
end
function dict_kernel!(du, u, p::DictP, t)
    @inbounds du[1] = -p.d[:k] * u[1]
    return nothing
end

@testset "ODE (OpaqueRef): numerical agreement with baseline" begin
    u0    = [1.0, 0.5]
    tspan = (0.0, 5.0)
    p     = CoeffP([1.0, 0.5, 0.25], 0.1)

    prob_plain  = ODEProblem(coeffs_plain!,                              u0, tspan, p)
    prob_opaque = ODEProblem(OpaqueRefRHS(CoeffP, coeffs_kernel!),       u0, tspan, pack_any(p))

    sol_p = solve(prob_plain,  Tsit5(), reltol = 1e-10, abstol = 1e-10)
    sol_o = solve(prob_opaque, Tsit5(), reltol = 1e-10, abstol = 1e-10)

    @test sol_p.t      ≈ sol_o.t      atol = 1e-9
    @test sol_p.u[end] ≈ sol_o.u[end] atol = 1e-9
end

@testset "ODE (OpaqueRef): prob.p uniform across non-isbits payloads" begin
    prob_a = ODEProblem(OpaqueRefRHS(CoeffP, coeffs_kernel!), [1.0, 0.5], (0.0, 1.0),
                        pack_any(CoeffP([1.0, 0.5, 0.25], 0.1)))
    prob_b = ODEProblem(OpaqueRefRHS(DictP,  dict_kernel!),   [1.0],      (0.0, 1.0),
                        pack_any(DictP(Dict(:k => 0.5))))

    @test typeof(prob_a.p) === typeof(prob_b.p) === OpaqueRef
end

@testset "ODE (OpaqueRef): rhs call is alloc-free" begin
    rhs = OpaqueRefRHS(CoeffP, coeffs_kernel!)
    op  = pack_any(CoeffP([1.0, 0.5, 0.25], 0.1))
    u   = [1.0, 0.5]
    du  = zero(u)
    rhs(du, u, op, 0.0)
    @test 0 == @allocated rhs(du, u, op, 0.0)
end

@testset "ODE (OpaqueRef): mutating the payload mid-flight propagates" begin
    u0    = [1.0, 0.5]
    tspan = (0.0, 1.0)
    p     = CoeffP([1.0, 0.5, 0.25], 0.1)
    op    = pack_any(p)
    prob  = ODEProblem(OpaqueRefRHS(CoeffP, coeffs_kernel!), u0, tspan, op)

    sol1 = solve(prob, Tsit5(), reltol = 1e-10, abstol = 1e-10)

    # Mutate the underlying vector (NOT the wrapper) — same OpaqueRef, new behavior.
    p.a[1] = 5.0
    sol2 = solve(prob, Tsit5(), reltol = 1e-10, abstol = 1e-10)
    @test sol1.u[end] != sol2.u[end]

    # Baseline with the mutated parameter.
    prob_baseline = ODEProblem(coeffs_plain!, u0, tspan, p)
    sol_base = solve(prob_baseline, Tsit5(), reltol = 1e-10, abstol = 1e-10)
    @test sol2.u[end] ≈ sol_base.u[end] atol = 1e-9
end
