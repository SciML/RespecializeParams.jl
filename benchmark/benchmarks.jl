using RespecializeParams, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

struct PendulumP
    L::Float64
    m::Float64
end
struct LotkaP
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
end

p = LotkaP(1.5, 1.0, 3.0, 1.0)
op = pack(p)

# =============================================================================
# pack / unpack — the core specialization-erasure API
# =============================================================================

SUITE["pack"] = BenchmarkGroup()

SUITE["pack"]["pack_struct"] = @benchmarkable pack($p)
SUITE["pack"]["pack_tuple"] = @benchmarkable pack((a = 1.0, b = 2.0, c = 3.5))
SUITE["pack"]["pack_tuple3"] = @benchmarkable pack((1.0, 2.0, 3.0, 4.0, 5.0))

SUITE["unpack"] = BenchmarkGroup()

SUITE["unpack"]["unpack"] = @benchmarkable unpack($op, LotkaP)
SUITE["unpack"]["unsafe_unpack"] = @benchmarkable unsafe_unpack($op, LotkaP)
SUITE["unpack"]["unpack_checked"] = @benchmarkable unpack_checked($op, LotkaP)

# =============================================================================
# RHS evaluation through opaque params (the representative workload: an ODE
# right-hand side dispatched on a type-erased parameter object)
# =============================================================================

function rhs!(du, u, op::OpaqueParams, t)
    p = unpack(op, LotkaP)
    du[1] = p.α * u[1] - p.β * u[1] * u[2]
    du[2] = -p.γ * u[2] + p.δ * u[1] * u[2]
    return nothing
end

du = zeros(2)
u = [1.0, 1.0]

SUITE["rhs"] = BenchmarkGroup()
SUITE["rhs"]["call"] = @benchmarkable rhs!($du, $u, $op, 0.0)
