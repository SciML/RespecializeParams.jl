using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "RespecializeParams" begin
    if GROUP == "All" || GROUP == "Core"
        @testset "core (OpaqueParams)" begin
            include("test_core.jl")
        end
        @testset "OpaqueRef" begin
            include("test_ref.jl")
        end
        @testset "OrdinaryDiffEq (OpaqueParams)" begin
            include("test_ode.jl")
        end
        @testset "OrdinaryDiffEq (OpaqueRef)" begin
            include("test_ode_ref.jl")
        end
        @testset "NonlinearSolve" begin
            include("test_nls.jl")
        end
    end

    if GROUP == "All" || GROUP == "QA"
        @testset "QA" begin
            include("qa/qa.jl")
        end
    end
end
