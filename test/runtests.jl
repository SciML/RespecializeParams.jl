using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@testset "RespecializeParams" begin
    if GROUP == "All" || GROUP == "Core"
        @safetestset "core (OpaqueParams)" begin
            include("test_core.jl")
        end
        @safetestset "OpaqueRef" begin
            include("test_ref.jl")
        end
        @safetestset "OrdinaryDiffEq (OpaqueParams)" begin
            include("test_ode.jl")
        end
        @safetestset "OrdinaryDiffEq (OpaqueRef)" begin
            include("test_ode_ref.jl")
        end
        @safetestset "NonlinearSolve" begin
            include("test_nls.jl")
        end
    end

    if GROUP == "All" || GROUP == "QA"
        @safetestset "QA" begin
            include("qa/qa.jl")
        end
    end
end
