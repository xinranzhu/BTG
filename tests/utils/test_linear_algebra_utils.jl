using Test
using Revise
include("../../src/Utils/linear_algebra_utils.jl")
using .linear_algebra_utils
import .linear_algebra_utils: diag

@testset "diag" begin
    x = rand(2, 2)
    @test diag(x) == [x[1, 1], x[2, 2]]
end
