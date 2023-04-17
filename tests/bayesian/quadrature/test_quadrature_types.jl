using Test

include("../../../src/bayesian/quadrature//quadrature.jl")
include("../../../src/core/parameters.jl")

parameters = [Parameter("a"), Parameter("b")]
#q1 = quad_info(Main.QuadratureTypes.Parameters.parameter("a"))
typeof(parameters[1])
q2 = QuadInfo(parameters[2])

@testset "Example 1" begin
    @test typeof(parameters[1]) == Parameter
    @test typeof(parameters[2]) == Parameter
    @test getrange(parameters[1]) == reshape([0, 1], 1, 2)
end
