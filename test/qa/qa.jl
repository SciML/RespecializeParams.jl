using RespecializeParams
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(RespecializeParams)
end

@testset "JET" begin
    JET.test_package(RespecializeParams; target_defined_modules = true)
end
