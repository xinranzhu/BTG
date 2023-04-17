include("../../src/transforms/registry.jl")
include("../../src/utils/derivative/derivative_checker.jl")
using Test
# using Plots


#short time test for build_input_dict
t = Affine() # should be nothing
d = Dict("a"=>1.02, "b" => 2.0)
@elapsed transform_dict = build_input_dict(t, d)
@elapsed transform_dict = build_input_dict_original(t, d)
#@test transform_dict["a"] == 1.02
#@test transform_dict["b"] == 2.0

@testset "empty initializer" begin
    t = SinhArcSinh()
    @test typeof(evaluate(t, 1.0, 1, 2))<:Real
end

#%%
@testset "partial initializer dict" begin
    t = SinhArcSinh(Dict("a"=>1))
    @test t.PARAMETER_DICT["a"] == 1
    @test t.PARAMETER_DICT["b"] == nothing
    @test t.a == 1
    @test t.b == nothing
    @test prod(t.names == ["a", "b"]) == 1

    t = SinhArcSinh(Dict("b"=>1))
    @test t.PARAMETER_DICT["a"] == nothing
    @test t.PARAMETER_DICT["b"] == 1
    @test t.a == nothing
    @test t.b == 1
    @test prod(t.names == ["a", "b"]) == 1

end

#%%
@testset "getArgs" begin
    t = SinhArcSinh(Dict("a"=>1, "b"=>2))
    d = Dict("a"=>100, "b"=>3)
    (a, b) = getArgs(t, d)
    @test (a, b) == (100, 3)
end
#%% check derivatives

@testset "SinhArcSinh derivative " begin
    t = SinhArcSinh(1, 2)
    my_f = x -> map(y -> evaluate(t, y), x)
    my_df = x -> map(y -> evaluate_derivative_x(t, y), x)

    my_f(5)
    my_df(5)

    (a, b, c, d) = checkDerivative(my_f, my_df, 2.0)
    d
    @test d[1] ≈ 2 rtol = .001 #coefficient of x approx 2
end
#%%

@testset "test derivatives wrt hyperparams" begin
    t = SinhArcSinh()
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
    @test r4.coeffs[2] ≈ 2 rtol = .01
end

#%%
@testset "test derivatives wrt hyperparams, second derivatives" begin
    t = SinhArcSinh()
    function evaluate_trans(arr)
        a = arr[1]
        b = arr[2]
        dict = Dict("a"=>a, "b"=>b)
        y = 1.0
        return evaluate_derivative_x(t, dict, y)
    end
    function evaluate_trans_deriv(arr)
        a = arr[1]
        b = arr[2]
        dict = Dict("a"=>a, "b"=>b)
        y = 1.0
        out_deriv_params = evaluate_derivative_x_hyperparams(t, dict, y)
        # out_deriv_y = evaluate_derivative_x(t, dict, y)
        return [out_deriv_params["a"], out_deriv_params["b"]]
    end
    arr = 2*rand(2)
    (r1, r2, r3, r4) = checkDerivative(evaluate_trans, evaluate_trans_deriv, arr)
    @show r4
    @test r4.coeffs[2] ≈ 2 rtol = 0.1
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
