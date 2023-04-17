using Test
using LinearAlgebra

include("../../src/kernels/rbf.jl")
include("../../src/kernels/Utils/utils.jl")
include("../../src/utils/linear_algebra_utils.jl")
import .linear_algebra_utils: diag

@testset "initialization" begin
    x1 = [1.0;2;3]
    x2 = [1.1;2.2;3.3]
    θ = 1.0
    rbf = RBFKernelModule(Dict("θ" => θ))
    @test prod(rbf.names .== ["θ", "var", "noise_var"]) == 1
    @test rbf.θ == θ
    @test rbf.var == nothing
    @test rbf.noise_var == nothing
    rbf_dict = rbf.PARAMETER_DICT
    @test rbf_dict["θ"] == θ
    @test rbf_dict["var"] == nothing
    @test rbf_dict["noise_var"] == nothing


    θ = [1.0;2.0;3.0]
    rbf = RBFKernelModule(Dict("θ" => θ))
    @test prod(rbf.θ .== θ)
    @test rbf.var == nothing
    @test rbf.noise_var == nothing
    rbf_dict = rbf.PARAMETER_DICT
    @test prod(rbf_dict["θ"] == θ)
    @test rbf_dict["var"] == nothing
    @test rbf_dict["noise_var"] == nothing


    rbf = RBFKernelModule(Dict("θ" => θ, "var" => 3.14))
    @test prod(rbf.θ == θ)
    @test rbf.var == 3.14
    @test rbf.noise_var == nothing
    rbf_dict = rbf.PARAMETER_DICT
    @test prod(rbf_dict["θ"] == θ)
    @test rbf_dict["var"] == 3.14
    @test rbf_dict["noise_var"] == nothing

end

@testset "no length scale" begin
    x1 = [1.0;2;3]
    x2 = [1.1;2.2;3.3]
    rbf = RBFKernelModule(Dict("θ" => 1.0, "var"=>1.0, "noise_var"=>0.0))
    res = evaluate(rbf, x1, x2)
    @test size(res) == (3, 3)
    @test size(diag(res)) == (3,)
    @test size(evaluate(rbf, x1, x1)) == (3, 3)
    @test size(evaluate(rbf, x1, x2)) == (3, 3)
    @test norm(diag(evaluate(rbf, x1, x1)) - [1.0, 1.0, 1.0])<1e-6
end

@testset "single length scale" begin
    θ = 1.0
    rbf = RBFKernelModule(Dict("θ" => θ, "var"=>1.0, "noise_var"=>0.0))
    x1 = [1.0;2;3]
    x2 = [1.1;2.2;3.3]
    res = evaluate(rbf, x1, x2)
    @test res[1, 1] ≈ 0.99501248 atol = 1e-4
    @test res[1, 2] ≈ 0.486752 atol = 1e-4
end

@testset "multiple length scales" begin
    θ = [1.0, 2.0]
    rbf = RBFKernelModule(Dict("θ" => θ, "var"=>1.0, "noise_var"=>0.0))
    x1 = [1.0 2.0; 2.0 3.0; 3.0 4.0]
    x2 = x1 .+ .1
    res = evaluate(rbf, x1, x2)
    @test res[1, 1] ≈ 0.9851119 atol = 1e-4
    res = evaluate(rbf, x1, x1)
    @test norm(diag(res) - [1, 1, 1]) < 1e-6
        #@test res[1, 2] ≈ 0.486752 atol = 1e-4
end

@testset "single length scales, derivative or kernel matrix" begin
    x1 = [1.0;2;3]
    x2 = [1.1;2.2;3.3]
    function my_f(arr)
        rbf = RBFKernelModule(Dict("θ" => arr, "var"=>1.0, "noise_var"=>0.0))
        res = evaluate(rbf, x1, x2)
        return res[:]
    end
    function my_df(arr)
        rbf = RBFKernelModule(Dict("θ" => arr, "var"=>1.0, "noise_var"=>0.0))
        res_deriv = evaluate_derivative_θ(rbf, x1, x2)
        return res_deriv[:]
    end
    (r1, r2, r3, r4) = checkDerivative(my_f, my_df, rand())
    @test abs(r4[1] - 2) < 0.1
end




@testset "multi-length scales, derivative or kernel matrix" begin
    x1 = [1.0  2.0;
            2.0 2.3;
            3.1  4.2 ]
    x2 = [0.2  2.4;
            1.2 2.9;
            1.1  10.2]

    # length(arr) = 1, arr = theta1
    function my_f1(arr)
        rbf = RBFKernelModule(Dict("θ" => [float(arr[1]), 0.3], "var"=>1.0, "noise_var"=>0.0))
        res = evaluate(rbf, x1, x2) # 3*3 matrix
        # @show arr
        # @show size(res[:])
        return res[:]
    end

    function my_df1(arr)
        rbf = RBFKernelModule(Dict("θ" => [float(arr[1]), 0.3], "var"=>1.0, "noise_var"=>0.0))
        res_deriv_set = evaluate_derivative_θ(rbf, x1, x2) #  3*3 matrix
        # @show arr
        # @show size(res_deriv_set[1][:])
        return res_deriv_set[1][:]
    end

    # length(arr) = 1, arr = theta2
    function my_f2(arr)
        rbf = RBFKernelModule(Dict("θ" => [0.2, float(arr[1])], "var"=>1.0, "noise_var"=>0.0))
        res = evaluate(rbf, x1, x2) # 3*3 matrix
        return res[:]
    end

    function my_df2(arr)
        rbf = RBFKernelModule(Dict("θ" => [0.2, float(arr[1])], "var"=>1.0, "noise_var"=>0.0))
        res_deriv_set = evaluate_derivative_θ(rbf, x1, x2) #  3*3 matrix
        return res_deriv_set[2][:]
    end

    (r1, r2, r3, r4) = checkDerivative(my_f1, my_df1, rand())
    @test abs(r4[1] - 2) < 0.1
    (r1, r2, r3, r4) = checkDerivative(my_f2, my_df2, rand())
    @test abs(r4[1] - 2) < 0.1

end
#%%
@testset "test derivative w.r.t position 2D θ" begin
kernel = RBFKernelModule([1.0, 3.0], 2.0, 3.0)
a = reshape([1 2.0], 1, 2)
b = [3 5; 3 4; 1 2.0]
@show a
@show b
a .- b
f(a) = evaluate(kernel, reshape(a, 1, length(a)), b)[:]
df(a) =  evaluate_derivative_a(kernel, reshape(a, 1, length(a)), b)
(_, _ ,_ , r4) = checkDerivative(f, df, [1, 2.0])
@test abs(r4[1] - 2)<=0.1
end

@testset "test derivative w.r.t position 1D θ" begin
    kernel = RBFKernelModule(1.0, 2.0, 3.0)
    a = reshape([1.0], 1, 1)
    b = reshape([3; 3; 1.2], 3, 1)
    evaluate_derivative_a(kernel, a, b)
    f(a) = evaluate(kernel, reshape(a, 1, length(a)), b)[:]
    df(a) =  evaluate_derivative_a(kernel, reshape(a, 1, length(a)), b)
    (_, _ ,_ , r4) = checkDerivative(f, df, reshape([1.5], 1, 1))
    @test abs(r4[1] - 2)<=0.1
end
