
include("../../src/transforms/registry.jl")
using Test


@testset "test initialization and evaluation" begin
    t = Id()
    @test t.names == []
    @test length(keys(t.PARAMETER_DICT)) == 0
    @test evaluate(t, 0.9) == 0.9
    @test prod(evaluate(t, [0.9, 1.1]) == [0.9, 1.1]) == 1
end

@testset "test derivative" begin
    t = Id()
    y = rand()
    @test evaluate_derivative_x(t, Dict(), y) == 1

    deriv_dict = evaluate_derivative_hyperparams(t, Dict(), y)
    @test deriv_dict == Dict()

    deriv_dict = evaluate_derivative_x_hyperparams(t, Dict(), y)
    @test deriv_dict == Dict()

end


@testset "build input dict" begin
    # should return empty dict
    t = Id()
    dict = Dict("var"=>1.0, "noise_var"=>0.01)
    @test build_input_dict(t, dict) == Dict()

end
