include("../../src/transforms/registry.jl")
include("../../src/utils/derivative/derivative_checker.jl")
using Test
# using Plots

@testset "empty initializer" begin
    t = ArcSinh()
    @test typeof(evaluate(t, 1.0, 1, 1, 1, 2))<:Real
end

#%%
@testset "partial initializer dict" begin
    t = ArcSinh(Dict("a"=>1))
    @test t.a == 1
    @test t.b == nothing
    @test t.c == nothing
    @test t.d == nothing

    t = ArcSinh(Dict("c"=>1, "d"=>2))
    @test t.a == nothing
    @test t.b == nothing
    @test t.c == 1
    @test t.d == 2
end

#%%
#@testset "getArgs" begin
t = ArcSinh(Dict("a"=>1, "d"=>4))
d = Dict("a"=>100, "b"=>2, "c"=>3)
(a, b, c, d) = getArgs(t, d)
@test (a, b, c, d) == (100, 2, 3, 4)

t = ArcSinh(Dict("a"=>1, "d"=>4))
d = Dict("a"=>100, "b"=>200, "c"=>300, "d"=>400)
@elapsed (a, b, c, e) = getArgs(t, d)
@elapsed (a, b, c, e) = getArgs_original(t, d)
@test (a, b, c, e) == (100, 200, 300, 400)
#end
#%% check derivatives

@testset "Arcsinh derivative " begin
    t = ArcSinh(1, 2, 3, 4)
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
    t = ArcSinh()
    function evaluate_trans(arr)
        a = 1.0; b = 0.1;  c = arr[1]; d = arr[2]
        dict = Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d)
        y = 1.0
        return evaluate(t, dict, y)
    end
    function evaluate_trans_deriv(arr)
        a = 1.0; b = 0.1;  c = arr[1]; d = arr[2]
        dict = Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d)
        y = 1.0
        out_deriv_params = evaluate_derivative_hyperparams(t, dict, y)
        # @show out_deriv_params
        return [out_deriv_params["c"], out_deriv_params["d"]]
    end
    arr = 2*rand(2)
    evaluate_trans(arr)
    evaluate_trans_deriv(arr)

    #evaluate_trans_deriv(arr)
    (r1, r2, r3, r4) = checkDerivative(evaluate_trans, evaluate_trans_deriv, arr)
    @show r4
    @test r4.coeffs[2] ≈ 2 rtol = .01

end
@testset "test derivatives wrt x and hyperparams" begin
    t = ArcSinh()
    function evaluate_trans(arr)
        a = 1.0; b = 0.1;  c = arr[1]; d = arr[2]
        dict = Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d)
        y = 1.0
        return evaluate_derivative_x(t, dict, y)
    end
    function evaluate_trans_deriv(arr)
        a = 1.0; b = 0.1;  c = arr[1]; d = arr[2]
        dict = Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d)
        y = 1.0
        out_deriv_params = evaluate_derivative_x_hyperparams(t, dict, y)
        # @show out_deriv_params
        return [out_deriv_params["c"], out_deriv_params["d"]]
    end
    arr = 2*rand(2)
    (r1, r2, r3, r4) = checkDerivative(evaluate_trans, evaluate_trans_deriv, arr)
    @show r4
    @test r4.coeffs[2] ≈ 2 rtol = .01

end



@testset "test build input dict" begin
    # the transform has no default value, d has all it's needed
    t = ArcSinh() # should be nothing
    d = Dict("a"=>1.02, "b" => 2.0, "c"=>7., "d"=>6.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 1.02
    @test transform_dict["b"] == 2.0
    @test transform_dict["c"] == 7.0
    @test transform_dict["d"] == 6.0

    # the transform has no default value, d has all it's needed and some other params too
    t = ArcSinh() # should be nothing
    d = Dict("a"=>1.02, "b" => 2.0, "var" => 1.232, "c"=>7., "d"=>6.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 1.02
    @test transform_dict["b"] == 2.0
    @test transform_dict["c"] == 7.0
    @test transform_dict["d"] == 6.0

    # the transform has partial default value, d has the complementary
    t = ArcSinh(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "c"=>7., "d"=>6.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 2.0
    @test transform_dict["c"] == 7.0
    @test transform_dict["d"] == 6.0

    # the transform has partial default value, d has the complementary and some other params too
    t = ArcSinh(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "var"=>1000., "c"=>7., "d"=>6.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 2.0
    @test transform_dict["c"] == 7.0
    @test transform_dict["d"] == 6.0

    # the transform has partial dafault value, d has the complementray and also wants to overwrite the default
    t = ArcSinh(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "a" => 999., "c"=>5., "d"=>6.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 2.0
    @test transform_dict["c"] == 5.0
    @test transform_dict["d"] == 6.0

    # the transform has partial dafault value, d has the complementray and also wants to cover the default, and has some other params too
    t = ArcSinh(Dict("a"=>0.11))
    # @show t
    d = Dict("b" => 2.0, "var"=>1000., "a" => 999., "c"=>5.6, "d"=>4.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 2.0
    @test transform_dict["c"] == 5.6
    @test transform_dict["d"] == 4.0

    # the transform has full params by default, d wants to overwrites some of them
    t = ArcSinh(Dict("a"=>0.11, "b" =>8.7, "c"=>7., "d"=>9.))
    # @show t
    d = Dict("a" => 999., "b"=>0.0)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 0.0
    @test transform_dict["c"] == 7.0
    @test transform_dict["d"] == 9.0

    # the transform has full params by default, d wants to overwrites some of them, d also has some other params
    t = ArcSinh(Dict("a"=>0.11, "b" =>8.7, "c"=>7., "d"=>9.))
    # @show t
    d = Dict("a" => 999., "b"=>0.0, "var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 0.0
    @test transform_dict["c"] == 7.0
    @test transform_dict["d"] == 9.0

    # the transform has full params by default, d wants to overwrites all of them
    t = ArcSinh(Dict("a"=>0.11, "b" =>8.7, "c"=>7., "d"=>9.))
    # @show t
    d = Dict("a" => 999., "b"=>888., "c"=>0.9, "d"=>9.1)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 888
    @test transform_dict["c"] == 0.9
    @test transform_dict["d"] == 9.1

    # the transform has full params by default, d wants to overwrites all of them, d also has some other params
    t = ArcSinh(Dict("a"=>0.11, "b" =>8.7, "c"=>7., "d"=>9.))
    # @show t
    d = Dict("a" => 999., "b"=>888., "c"=>0.9, "d"=>9.1, "var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 999.
    @test transform_dict["b"] == 888
    @test transform_dict["c"] == 0.9
    @test transform_dict["d"] == 9.1

    # the transform has full params by default, d is empty
    t = ArcSinh(Dict("a"=>0.11, "b" =>8.7, "c"=>7., "d"=>9.))
    # @show t
    d = Dict()
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 8.7
    @test transform_dict["c"] == 7.0
    @test transform_dict["d"] == 9.0

    # the transform has full params by default, d only has other unrelated parameters
    t = ArcSinh(Dict("a"=>0.11, "b" =>8.7, "c"=>7., "d"=>9.))
    # @show t
    d = Dict("var"=>1000.)
    transform_dict = build_input_dict(t, d)
    @test transform_dict["a"] == 0.11
    @test transform_dict["b"] == 8.7
    @test transform_dict["c"] == 7.0
    @test transform_dict["d"] == 9.0

end
