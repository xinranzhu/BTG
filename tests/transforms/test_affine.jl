include("../../src/transforms/registry.jl")
include("../../src/utils/derivative/derivative_checker.jl")
using Test

# %%
@testset "empty initializer" begin
    t = Affine()
    @test typeof(evaluate(t, 1.0, 1, 2))<:Real
end

#%%
@testset "partial initializer dict" begin
    t = Affine(Dict("a"=>1))
    @test t.a == 1
    @test t.b == nothing

    t = Affine(Dict("b"=>1))
    @test t.a == nothing
    @test t.b == 1
end

#%%
@testset "getArgs" begin
    t = Affine(Dict("a"=>1, "b"=>2))
    d = Dict("a"=>100, "b"=>3)
    (a, b) = getArgs(t, d)
    @test (a, b) == (100, 3)
end


@testset "test derivatives" begin
    t = Affine()
    function evaluate_trans(arr)
        a = arr[1]
        b = arr[2]
        dict = Dict("a"=>a, "b"=>b)
        y = arr[3]
        return evaluate(t, dict, y)
    end
    function evaluate_trans_deriv(arr)
        a = arr[1]
        b = arr[2]
        dict = Dict("a"=>a, "b"=>b)
        y = arr[3]
        out_deriv_params = evaluate_derivative_hyperparams(t, dict, y)
        out_deriv_y = evaluate_derivative_x(t, dict, y)
        return [out_deriv_params["a"], out_deriv_params["b"], out_deriv_y]
    end
    arr = 2*rand(3)
    (r1, r2, r3, r4) = checkDerivative(evaluate_trans, evaluate_trans_deriv, arr)
    @show r4
    @test r4.coeffs[2] ≈ 2 rtol = .0001
end


@testset "test build input dict" begin
    # the transform has no default value, d has all it's needed
    t = Affine() # should be nothing
    d = Dict("a"=>1.02, "b" => 2.0)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 1.02
    @test transform_dict["b"] == 2.0

    # the transform has no default value, d has all it's needed and some other params too
    t = Affine() # should be nothing
    d = Dict("a"=>1.02, "b" => 2.0, "var" => 1.232)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 1.02
    @test transform_dict["b"] == 2.0

    # the transform has partial default value, d has the complementary
    t = Affine(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 2.0

    # the transform has partial default value, d has the complementary and some other params too
    t = Affine(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 2.0

    # the transform has partial dafault value, d has the complementray and also wants to cover the default
    t = Affine(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "a" => 999.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 2.0

    # the transform has partial dafault value, d has the complementray and also wants to cover the default, and has some other params too
    t = Affine(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "var"=>1000., "a" => 999.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 2.0

    # the transform has full params by default, d wants to overwrites some of them
    t = Affine(Dict("a"=>0.11, "b" =>8.7))
    # @show t
    d = Dict("a" => 999.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 8.7

    # the transform has full params by default, d wants to overwrites some of them, d also has some other params
    t = Affine(Dict("a"=>0.11, "b" =>8.7))
    # @show t
    d = Dict("a" => 999., "var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 8.7

    # the transform has full params by default, d wants to overwrites all of them
    t = Affine(Dict("a"=>0.11, "b" =>8.7))
    # @show t
    d = Dict("a" => 999., "b"=>888.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 888

    # the transform has full params by default, d wants to overwrites all of them, d also has some other params
    t = Affine(Dict("a"=>0.11, "b" =>8.7))
    # @show t
    d = Dict("a" => 999., "b"=>888., "var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 888

    # the transform has full params by default, d is empty
    t = Affine(Dict("a"=>0.11, "b" =>8.7))
    # @show t
    d = Dict()
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 8.7

    # the transform has full params by default, d only has other unrelated parameters
    t = Affine(Dict("a"=>0.11, "b" =>8.7))
    # @show t
    d = Dict("var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 8.7

end
