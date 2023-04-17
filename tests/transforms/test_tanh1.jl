include("../../src/transforms/registry.jl")
include("../../src/utils/derivative/derivative_checker.jl")
using Test

# %%
@testset "empty initializer" begin
    t = Tanh1()
    @show evaluate(t, 1.0, 1, 0.3, 2)
    @test typeof(evaluate(t, 1.0, 1, 0.3, 2))<:Real
end

#%%
@testset "partial initializer dict" begin
    t = Tanh1(Dict("a"=>1))
    @test t.a == 1
    @test t.b == nothing
    @test t.d == nothing

    t = Tanh1(Dict("b"=>1))
    @test t.a == nothing
    @test t.b == 1
    @test t.d == nothing
end

#%%
@testset "getArgs" begin
    t = Tanh1(Dict("a"=>1, "b"=>2))
    d = Dict("a"=>100, "b"=>3, "d"=>2)
    (a, b, c) = getArgs(t, d)
    @test (a, b, c) == (100, 3, 2)
end


@testset "test derivatives" begin
    t = Tanh1()
    function evaluate_trans(arr)
        a = arr[1]
        b = arr[2]
        c = arr[3]
        dict = Dict("a"=>a, "b"=>b, "d"=>c)
        y = arr[4]
        return evaluate(t, dict, y)
    end
    function evaluate_trans_deriv(arr)
        a = arr[1]
        b = arr[2]
        c = arr[3]
        dict = Dict("a"=>a, "b"=>b, "d"=>c)
        y = arr[4]
        out_deriv_params = evaluate_derivative_hyperparams(t, dict, y)
        out_deriv_y = evaluate_derivative_x(t, dict, y)
        return [out_deriv_params["a"], out_deriv_params["b"], out_deriv_params["d"], out_deriv_y]
    end
    arr = 2*rand(4)
    (r1, r2, r3, r4) = checkDerivative(evaluate_trans, evaluate_trans_deriv, arr)
    @show r4
    @test r4.coeffs[2] ≈ 2 rtol = .1
end

@testset "test second derivatives" begin
    t = Tanh1()
    # a = rand()
    # b = rand()
    # c = rand()
    # y = rand()
    function evaluate_trans(arr)
        a = arr[1]
        b = arr[2]
        c = arr[3]
        y = arr[4]
        dict = Dict("a"=>a, "b"=>b, "d"=>c)
        return evaluate_derivative_x(t, dict, y)
    end
    function evaluate_trans_deriv(arr)
        a = arr[1]
        b = arr[2]
        c = arr[3]
        dict = Dict("a"=>a, "b"=>b, "d"=>c)
        y = arr[4]
        out_deriv_params = evaluate_derivative_x_hyperparams(t, dict, y)
        out_deriv_y = evaluate_derivative_x2(t, dict, y)
        return [out_deriv_params["a"], out_deriv_params["b"], out_deriv_params["d"], out_deriv_y]
        # return [out_deriv_params["a"], out_deriv_params["b"]]
    end
    arr = 2*rand(4)
    (r1, r2, r3, r4) = checkDerivative(evaluate_trans, evaluate_trans_deriv, arr)
    @show r4
    @test r4.coeffs[2] ≈ 2 rtol = .1
end


@testset "test inverse" begin
    d = Dict("a"=>1.02, "b" => 2.0, "d"=>4.0)
    t = Tanh1() 
    x = rand()
    y = evaluate_inverse(t, d, x)
    x_test = evaluate(t, d, y)
    @test abs(x - x_test) < 1e-3
end

@testset "test build input dict" begin
    # the transform has no default value, d has all it's needed
    t = Tanh1() # should be nothing
    d = Dict("a"=>1.02, "b" => 2.0, "d"=>4.0)
    transform_dict = build_input_dict(t, d)
    @show transform_dict
    @test transform_dict["a"] == 1.02
    @test transform_dict["b"] == 2.0
    @test transform_dict["d"] == 4.0

    # the transform has no default value, d has all it's needed and some other params too
    t = Tanh1() # should be nothing
    d = Dict("a"=>1.02, "b" => 2.0, "d"=>4.0, "var" => 1.232)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 1.02
    @test transform_dict["b"] == 2.0
    @test transform_dict["d"] == 4.0

    # the transform has partial default value, d has the complementary
    t = Tanh1(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "d"=>3.4)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 2.0
    @test transform_dict["d"] == 3.4

    # the transform has partial default value, d has the complementary and some other params too
    t = Tanh1(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0,  "d"=>3.4, "var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 2.0
    @test transform_dict["d"] == 3.4

    # the transform has partial dafault value, d has the complementray and also wants to cover the default
    t = Tanh1(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "a" => 999., "d"=>9.0)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 2.0
    @test transform_dict["d"] == 9.0

    # the transform has partial dafault value, d has the complementray and also wants to cover the default, and has some other params too
    t = Tanh1(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "var"=>1000., "a" => 999., "d"=>9.0)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 2.0
    @test transform_dict["d"] == 9.0

    # the transform has full params by default, d wants to overwrites some of them
    t = Tanh1(Dict("a"=>0.11, "b" =>8.7))
    # @show t
    d = Dict("a" => 999., "d"=>9.0)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 8.7
    @test transform_dict["d"] == 9.0


    # the transform has full params by default, d wants to overwrites some of them, d also has some other params
    t = Tanh1(Dict("a"=>0.11, "b" =>8.7, "d"=>9.0))
    # @show t
    d = Dict("a" => 999., "var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 8.7
    @test transform_dict["d"] == 9.0


    # the transform has full params by default, d wants to overwrites all of them
    t = Tanh1(Dict("a"=>0.11, "b" =>8.7, "d"=>9.0))
    # @show t
    d = Dict("a" => 999., "b"=>888., "d"=>1.0)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 888
    @test transform_dict["d"] == 1.0


    # the transform has full params by default, d wants to overwrites all of them, d also has some other params
    t = Tanh1(Dict("a"=>0.11, "b" =>8.7, "d"=>9.0))
    # @show t
    d = Dict("a" => 999., "b"=>888., "d"=>1.0, "var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 888
    @test transform_dict["d"] == 1.0

    # the transform has full params by default, d is empty
    t = Tanh1(Dict("a"=>0.11, "b" =>8.7, "d"=>1.0))
    # @show t
    d = Dict()
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 8.7
    @test transform_dict["d"] == 1.0

    # the transform has full params by default, d only has other unrelated parameters
    t = Tanh1(Dict("a"=>0.11, "b" =>8.7, "d"=>1.0))
    # @show t
    d = Dict("var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 8.7
    @test transform_dict["d"] == 1.0

end
