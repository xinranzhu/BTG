using Test
using Cubature #test by comparing to cubature results?
using LinearAlgebra
include("../../../src/BTG.jl")
#%%
zoo =
[
Parameter("a", 1, reshape([0, 1], 1, 2)),
Parameter("b", 1, reshape([1, 2], 1, 2)),
Parameter("c", 1, reshape([1.0, 1.5], 1, 2)),
Parameter("d", 2, reshape([1.2 1.3; 3.4 3.6], 2, 2)),
Parameter("e", 2, reshape([1.5 1.9; 3.1 3.2], 2, 2)),
]
#%%
quad_zoo =
[
    QuadType(3),
    QuadType(3),
    QuadType(2),
    QuadType(1),
    QuadType(3),
    QuadType(2),
]
#%%
parameters = zoo[1:2]
m = Dict(getname(parameters[i]) => QuadInfo(parameters[i], quad_type = quad_zoo[i], levels = 5, num_points = 4) for i = 1:length(parameters))
qs = QuadratureSpecs(parameters, m) #quadrature specs object
qd = QuadratureDomain(qs) #quadrature domain object

#%%
@testset "example1" begin
    f = x -> x["a"][1].^2+x["b"][1].^2 #f(x, y) = x^2+y^2
    res = integrate(f, qd)
    @test res ≈ 2.6666 atol=1e-3
end
