include("../../../src/BTG.jl")
using Test

#%%
@testset "test \"likelihood_fixed_train\" in \"build_normalized_weight_iterator\" can reformat inputs" begin
    my_transform = ComposedTransformation([Affine(1.0, 2.0), SinhArcSinh(4.0, 5.0)])
    kernel_module = RBFKernelModule()
    buffer = BufferDict()
    train = get_random_training_data()
    lookup = true
    function likelihood_fixed_train(d, buffer::BufferDict; log_scale=false, lookup = lookup) #TODO typecheck d
        d_copy = Dict()
        for k in keys(d)
            if length(d[k]) == 1
                d_copy[k] = d[k][1]
            else
                d_copy[k] = d[k]
            end
        end
        # @show d_copy
        reformatted_dict  = reformat(d_copy, my_transform)
        @show reformatted_dict
        return WarpedGPLikelihood(reformatted_dict, train, my_transform, kernel_module, buffer)
    end
    d = Dict("COMPOSED_TRANSFORM_PARAMETER_1"=>[1.9, 2.0], "COMPOSED_TRANSFORM_PARAMETER_2"=>[3.4, 5.2], "θ"=>3.4, "var"=>1.0, "noise_var"=>3.0)
    res = likelihood_fixed_train(d, BufferDict())
    @test typeof(res) <:Float64
end

#%%
@testset "1" begin
a = Parameter("a", 1, reshape([0, 1], 1, 2))
b = Parameter("b", 1, reshape([0, 1], 1, 2))
params =[a, b]
m = Dict("a"=>QuadInfo(a; quad_type =  QuadType(1), num_points = 4),
    "b"=>QuadInfo(b; quad_type =  QuadType(2), num_points = 4))

qs = QuadratureSpecs(params, m)
qd = QuadratureDomain(qs)

w = get_iterator(qd)
@test length(w) == 16
end

#%%

@testset "2" begin
    a = Parameter("a", 1, reshape([0, 1], 1, 2))
    b = Parameter("b", 1, reshape([0, 1], 1, 2))
    params =[a, b]
    m = Dict("a"=>QuadInfo(a; quad_type =  QuadType(0), num_points = 4),
        "b"=>QuadInfo(b; quad_type =  QuadType(0), num_points = 5))

    qs = QuadratureSpecs(params, m)
    qd = QuadratureDomain(qs)

    w = get_iterator(qd)
    @test length(w) == 25 #5 times 5 = 25
end

@testset "2" begin
    a = Parameter("a", 1, reshape([0, 1], 1, 2))
    b = Parameter("b", 1, reshape([0, 1], 1, 2))
    params =[a, b]
    m = Dict("a"=>QuadInfo(a; quad_type =  QuadType(0), num_points = 4),
        "b"=>QuadInfo(b; quad_type =  QuadType(1), num_points = 5))

    qs = QuadratureSpecs(params, m)
    qd = QuadratureDomain(qs)

    w = get_iterator(qd)
    @test length(w) == 20 #5 times 5 = 25
end
