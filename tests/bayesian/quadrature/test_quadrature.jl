using Test
using Cubature
using LinearAlgebra

include("../../../src/bayesian/quadrature/quadrature.jl")
include("../../../src/bayesian/quadrature/quadrature_types.jl")
include("../../../src/bayesian/quadrature/domain.jl")


#@testset "nodes_weights.jl" begin
## test init, 1D
range_param = reshape([0, 20], 1, 2)
dim = size(range_param)[1]
num_nodes = 11; levels = 5
a = Parameter("a", dim, range_param)
q = QuadType(0)
info = QuadInfo(a, quad_type = q)
nw = get_nodes_weights(q, info.range; num_nodes=num_nodes, levels=levels)
@test size(nw) == (num_nodes^(dim), 2)

## test init, 3D
range_param = [1 3; -2 -1; 8 10]
dim = size(range_param)[1]
num_nodes = 11; levels = 5
a = Parameter("a", dim, range_param)
q = QuadType(0)
info = QuadInfo(a, quad_type = q)
nw = get_nodes_weights(q, info.range; num_nodes=num_nodes, levels=levels)
@test size(nw) == (num_nodes^(dim), dim+1)


## test integrate, 1D
f = x -> sin(x) + x
range_param = reshape([-20, 20], 1, 2)
dim = size(range_param)[1]
a = Parameter("a", dim, range_param)
range_int = reshape([-1, 2], 1, 2)
res_true = hquadrature(f, range_int[1], range_int[2])[1]

## Gaussian
num_nodes = 40
q = QuadType(0)
info = QuadInfo(a, quad_type = q, num_points = num_nodes, range = range_int)
nw = get_nodes_weights(q, info.range; num_nodes=num_nodes)
res = dot(f.(nw[:,1]), nw[:, 2])
@test res ≈ res_true atol = 1e-3
## MonteCarlo
num_nodes = 10000
q = QuadType(1)
info = QuadInfo(a, quad_type = q, num_points = num_nodes, range = range_int)
nw = get_nodes_weights(q, info.range; num_nodes=num_nodes)
res = dot(f.(nw[:,1]), nw[:, 2])
@test res ≈ res_true atol = 1e-1
## QuasiMonteCarlo
num_nodes = 1000
q = QuadType(2)
info = QuadInfo(a, quad_type = q, num_points = num_nodes, range = range_int)
nw = get_nodes_weights(q, info.range; num_nodes=num_nodes)
res = dot(f.(nw[:,1]), nw[:, 2])
@test res ≈ res_true atol = 1e-3
## SparseGrid
q = QuadType(3)
info = QuadInfo(a, quad_type = q, num_points = num_nodes, range = range_int)
nw = get_nodes_weights(q, info.range)
res = dot(f.(nw[:,1]), nw[:, 2])
@test res ≈ res_true atol = 1e-3

# test integrate, 3D
g(x) = sin(x[1])*(x[2]+1) + sqrt(x[3])
range_param = [1 3; -2 -1; 8 10]
dim = size(range_param)[1]
a = Parameter("a", dim, range_param)
res_true = hcubature(g, range_param[:,1], range_param[:,2])[1]
## Gaussian
num_nodes = 40
q = QuadType(0)
info = QuadInfo(a, quad_type = q, num_points = num_nodes)
nw = get_nodes_weights(q, info.range; num_nodes=num_nodes)
res = dot([g(nw[i,1:dim]) for i in 1:size(nw,1)], nw[:, end])
@test res ≈ res_true atol = 1e-3
## MonteCarlo
num_nodes = 4000
q = QuadType(1)
info = QuadInfo(a, quad_type = q, num_points = num_nodes)
nw = get_nodes_weights(q, info.range; num_nodes=num_nodes)
res = dot([g(nw[i,1:dim]) for i in 1:size(nw,1)], nw[:, end])
@test res ≈ res_true atol = 1e-1
## QuasiMonteCarlo
num_nodes = 1000
q = QuadType(2)
info = QuadInfo(a, quad_type = q, num_points = num_nodes)
nw = get_nodes_weights(q, info.range; num_nodes=num_nodes)
res = dot([g(nw[i,1:dim]) for i in 1:size(nw,1)], nw[:, end])
@test res ≈ res_true atol = 1e-3
## SparseGrid
q = QuadType(3)
info = QuadInfo(a, quad_type = q, num_points = num_nodes)
nw = get_nodes_weights(q, info.range)
res = dot([g(nw[i,1:dim]) for i in 1:size(nw,1)], nw[:, end])
@test res ≈ res_true atol = 1e-3
#end

#@testset "test test" begin #initial testing for writing a test
#parameters = [Parameter("a", 1, reshape([1, 2], 1, 2)), Parameter("b", 1, reshape([2, 3], 1, 2))]
#x = merge(parameters)

#m = Dict("a"=> QuadInfo(parameters[1]), "b"=>QuadInfo(parameters[2]))
#qs = QuadratureSpecs(parameters, m)
#qd = QuadratureDomain()
#quad_specs = QuadratureSpecs()

#y = sort(collect(quad_specs.quad_info_dict), by = a -> getquadtype(a.second))
#filtered_specs = []
#for c in groupby(x -> getquadtype(x.second[1]), y)
#   filtered_specs.append(c)
#end
#print(filtered_specs)
#end


#@testset "domain.jl" begin
#    parameters = [Parameter("a", [1, 2]), Parameter("b", [2, 3])]
#    d = Dict( "a" =>  quad_info(parameters[1]), "b"=> quad_info(parameters[2]))
#    quad_specs = quadrature_specs(parameters, d)
#end

#@testset "quadrature_types.jl" begin
#    parameters = [Parameter("a", [1, 2]), Parameter("b", [2, 3])]
#    #q1 = quad_info(Main.QuadratureTypes.Parameters.parameter("a"))
###    q2 = quad_info(parameters[2])
#
#end
