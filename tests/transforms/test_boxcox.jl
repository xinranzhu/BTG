include("../../src/transforms/registry.jl")
include("../../src/utils/derivative/derivative_checker.jl")
using Test

# %%
@testset "empty initializer" begin
    t = BoxCox()
    @test typeof(evaluate(t, 1.0, 2))<:Real
end

#%%
@testset "complete initializer dict" begin
    t = BoxCox(Dict("λ"=>1))
    @show t
    @test t.λ == 1
end

# %%
@testset "getArgs" begin
    t = BoxCox(Dict("λ"=>1))
    d = Dict("λ"=>100)
    (λ) = getArgs(t, d)
    @test λ[1] == 100
end


@testset "test derivatives" begin
    t = BoxCox()
    function evaluate_trans(arr)
        λ = arr[1]
        dict = Dict("λ"=>λ)
        y = arr[2]
        return evaluate(t, dict, y)
    end
    function evaluate_trans_deriv(arr)
        λ = arr[1]
        dict = Dict("λ"=>λ)
        y = arr[2]
        out_deriv_params = evaluate_derivative_hyperparams(t, dict, y)
        out_deriv_y = evaluate_derivative_x(t, dict, y)
        return [out_deriv_params["λ"], out_deriv_y]
    end
    arr = 2*rand(2)
    (r1, r2, r3, r4) = checkDerivative(evaluate_trans, evaluate_trans_deriv, arr)
    @show r4
    @test r4.coeffs[2] ≈ 2 rtol = .1
end


@testset "test second derivatives" begin
    t = BoxCox()
    function evaluate_trans(arr)
        λ = arr[1]
        y = arr[2]
        dict = Dict("λ"=>λ)
        return evaluate_derivative_x(t, dict, y)
    end
    function evaluate_trans_deriv(arr)
        λ = arr[1]
        dict = Dict("λ"=>λ)
        y = arr[2]
        out_deriv_params = evaluate_derivative_x_hyperparams(t, dict, y)
        out_deriv_y = evaluate_derivative_x2(t, dict, y)
        return [out_deriv_params["λ"], out_deriv_y]
    end
    arr = 2*rand(2)
    (r1, r2, r3, r4) = checkDerivative(evaluate_trans, evaluate_trans_deriv, arr)
    @show r4
    @test r4.coeffs[2] ≈ 2 rtol = .1
end


@testset "test inverse" begin
    d = Dict("λ"=>0.6)
    t = BoxCox() 
    x = rand()
    y = evaluate_inverse(t, d, x)
    x_test = evaluate(t, d, y)
    @test abs(x - x_test) < 1e-3
end

@testset "test build input dict" begin
    # the transform has no default value, d has all it's needed
    t = BoxCox() # should be nothing
    d = Dict("λ"=>1.02)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["λ"] == 1.02

    # the transform has no default value, d has all it's needed and some other params too
    t = BoxCox() # should be nothing
    d = Dict("λ"=>1.02, "b" => 2.0, "var" => 1.232)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["λ"] == 1.02

    # the transform has partial default value, d has the complementary
    t = BoxCox(Dict("λ"=>0.11))
    # @show t
    d = Dict("λ" => 2.0)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["λ"] == 2.0

end
