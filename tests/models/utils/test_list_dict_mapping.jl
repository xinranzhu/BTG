using Random
using Test

include("../../../src/BTG.jl")

@testset "various dictionary, flattened, nested list conversions" begin
    parameter_names = [[["a", "b"], ["c"]]]
    d = Dict("COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>2.0, "b"=>4.0),Dict("c"=>5.0)])
    nested = dictionary_to_nested_list(parameter_names, d)
    @assert nested == [[[2.00, 4.00], [5.00]]]

    parameter_names = ["θ", [["a", "b"], ["c"]]]
    d = Dict("COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>2.0, "b"=>4.0),Dict("c"=>5.0)], "θ"=>3.5)
    nested = dictionary_to_nested_list(parameter_names, d)
    @test nested == [3.5,[[2.0, 4.0], [5.0]]]

    ww = flatten_nested_list(nested)
    @test ww == [3.5, 2.0, 4.0, 5.0]

    dict = flattened_list_to_dictionary(parameter_names, ww)
    @test dict == d

    gg = flattened_list_to_nested_list(parameter_names, ww)
    @test gg == [3.5, [[2.0, 4.0], [5.0]]]

    tt = dictionary_to_flattened_list(parameter_names, d)
    @test tt == ww

    oo = nested_list_to_dictionary(parameter_names, nested)
    @assert oo == d
end

@testset "test nested_list_to_dictionary, complete param info" begin
    parameter_names = ["θ", "var", Any[["a", "b", "c", "d"], ["a", "b"]]]
    arr =  [1.0e-5, 2.0e-3, 0.11, 0.12, 0.2, 0.3, 1.34, 100.]
    dict = nested_list_to_dictionary(parameter_names, arr)
    @show dict
    @test dict == Dict{Any,Any}("COMPOSED_TRANSFORM_DICTS" => [Dict{Any,Any}("c" => 0.2,"b" => 0.12,"a" => 0.11,"d" => 0.3), 
                                                                    Dict{Any,Any}("b" => 100.0,"a" => 1.34)],
                                    "θ" => 1.0e-5,"var" => 0.002)
end


@testset "test nested_list_to_dictionary, partial param info" begin
    #TODO
    # parameter_names = ["θ", "var", Any[["a", "b", "c", "d"], ["a", "b"]]]
    # arr =  [1.0e-5, 2.0e-3, 0.11, 0.12, 0.2, 0.3, 1.34, 100.]
    # dict = nested_list_to_dictionary(parameter_names, arr)
    # @show dict
    # @test dict == Dict{Any,Any}("COMPOSED_TRANSFORM_DICTS" =>  [Dict{Any,Any}("c" => 0.2,"b" => 0.12,"a" => 0.11,"d" => 0.3), 
    #                                                             Dict{Any,Any}("b" => 100.0,"a" => 1.34)],
    #                                 "θ" => 1.0e-5, "var" => 0.002)
end

@testset "test flattened_list_to_dictionary for multi-lengthscale, composed transform" begin
    parameter_names = [["θ1", "θ2"], "var", [["a", "b"], ["c"]]]
    d = Dict("θ"=>[0.8, 0.9], "var"=>0.0, "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>2.0, "b"=>4.0),Dict("c"=>5.0)] )
    flattened_list = [0.8, 0.9, 0.0, 2.0, 4.0, 5.0]
    dict = flattened_list_to_dictionary(parameter_names, flattened_list)
    @show dict
    @test dict == d
end

@testset "test flattened_list_to_dictionary for multi-lengthscale, single transform" begin
    parameter_names = [["θ1", "θ2"], "var", "a", "b", "c"]
    d = Dict("θ"=>[0.8, 0.9], "var"=>0.0, "a"=>2.0, "b"=>4.0, "c"=>5.0 )
    flattened_list = [0.8, 0.9, 0.0, 2.0, 4.0, 5.0]
    dict = flattened_list_to_dictionary(parameter_names, flattened_list)
    @show dict
    @test dict == d
end

@testset "test flattened_list_to_nested_list, multi-lengthscale, composed transform" begin
        parameter_names =  ["var",  [["a", "b", "c", "d"], ["a", "b"]], ["θ1", "θ2"]]
        arr = [1., 2., 3. , 4., 5., 6., 7., 8., 9.]
        final_list_true = [1., [[2., 3., 4., 5.], [6., 7.]], [8., 9.]]
        final_list = flattened_list_to_nested_list(parameter_names, arr)
        @show final_list
        @test prod(final_list_true .== final_list)
end

@testset "test flattened_list_to_nested_list, multi-lengthscale, single transform" begin
        parameter_names =  ["var",  "a", "b", ["θ1", "θ2"]]
        arr = [1., 2., 3. , 4., 5.]
        final_list_true = [1., 2., 3., [4., 5.]]
        final_list = flattened_list_to_nested_list(parameter_names, arr)
        @show final_list
        @test prod(final_list_true .== final_list)
end


@testset "dictionary_to_flattened_list, multidimensional theta" begin
        dict = Dict("θ" => [0.8, 0.9], "var"=>0.1,             
                "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>11.0, "b"=>12.0), Dict("c"=>13.0)])
        parameter_names =  ["var",  [["a", "b"], ["c"]], ["θ1", "θ2"]]
        arr_true = [0.1, 11., 12., 13., 0.8, 0.9]
        arr = dictionary_to_flattened_list(parameter_names, dict)
        
end