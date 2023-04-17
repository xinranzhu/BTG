
using Random
using Test

include("../../../src/BTG.jl")
t_affine = Affine(0.0, 1.0) 
t_sinharcsinh = SinhArcSinh(0.3, 1.2)
t_arcsinh = ArcSinh(1.234, 5.678, 2., 3.)

@testset "Test for one component only" begin
    T = [t_affine, t_sinharcsinh]
    transform = ComposedTransformation(T)
    parameter_names = ["θ", "var", "noise_var", "COMPOSED_TRANSFORM_PARAMETER_2"]
    lower_bound = [0.1, 0.2, 0.3, [0.4, 0.5]]
    upper_bound = [1.0, 2.0, 3.0, [4., 5.]];
    parameter_names_new, lower_new, upper_new = parse_input(parameter_names, lower_bound, upper_bound, transform)
    @test prod(lower_new .==  [0.1, 0.2, 0.3, 0.4, 0.5]) == 1
    @test prod(upper_new .== [1.0, 2.0, 3.0, 4., 5.]) == 1
    @test prod(parameter_names_new .== Any["θ", "var", "noise_var", Any[Any[], ["a", "b"]]]) == 1
end

@testset "Test for both two components" begin
    T = [t_arcsinh, t_sinharcsinh]
    transform = ComposedTransformation(T)
    parameter_names = ["θ", "var", "COMPOSED_TRANSFORM_PARAMETER_2", "COMPOSED_TRANSFORM_PARAMETER_1"]
    lower_bound = [0.1, 0.2,  [0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    upper_bound = [1.0, 2.0, [3., 4.], [5., 6., 7., 8.]];
    parameter_names_new, lower_new, upper_new = parse_input(parameter_names, lower_bound, upper_bound, transform)
    @test prod(lower_new .== [0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.3, 0.4]) == 1
    @test prod(upper_new .== [1.0, 2.0, 5., 6., 7., 8., 3., 4.]) == 1
    @test prod(parameter_names_new .== Any["θ", "var", Any[["a", "b", "c", "d"], ["a", "b"]]]) == 1
end

@testset "Test for multidimensional theta" begin
    T = [t_arcsinh, t_sinharcsinh]
    transform = ComposedTransformation(T)
    parameter_names = ["θ", "var", "COMPOSED_TRANSFORM_PARAMETER_2", "COMPOSED_TRANSFORM_PARAMETER_1"]
    lower_bound = [[0.1, 0.2], 0.3,  [0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]
    upper_bound = [[1.0, 2.0], 3.0, [4., 5.], [6., 7., 8., 9.]];
    parameter_names_new, lower_new, upper_new = parse_input(parameter_names, lower_bound, upper_bound, transform)
    # @show parameter_names_new
    # @show lower_new
    # @show upper_new
    @test prod(parameter_names_new .== ["var", Any[["a", "b", "c", "d"], ["a", "b"]], ["θ1", "θ2"]]) == 1
    @test prod(lower_new .== [0.3, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.1, 0.2]) == 1
    @test prod(upper_new .== [3.0, 6., 7., 8., 9., 4., 5., 1.0, 2.0]) == 1
end

@testset "Test for multidimensional theta, change order" begin
    T = [t_arcsinh, t_sinharcsinh]
    transform = ComposedTransformation(T)
    parameter_names = ["var", "θ", "COMPOSED_TRANSFORM_PARAMETER_2", "COMPOSED_TRANSFORM_PARAMETER_1"]
    lower_bound = [0.3, [0.1, 0.2], [0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]
    upper_bound = [3.0, [1.0, 2.0], [4., 5.], [6., 7., 8., 9.]];
    parameter_names_new, lower_new, upper_new = parse_input(parameter_names, lower_bound, upper_bound, transform)
    @test prod(parameter_names_new .== ["var", Any[["a", "b", "c", "d"], ["a", "b"]], ["θ1", "θ2"]]) == 1
    @test prod(lower_new .== [0.3, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.1, 0.2]) == 1
    @test prod(upper_new .== [3.0, 6., 7., 8., 9., 4., 5., 1, 2]) == 1
end

@testset "Test for multidimensional theta, single transformation" begin
    transform = t_sinharcsinh
    parameter_names = ["var", "θ",  "a", "b"]
    lower_bound = [0.3, [0.1, 0.2], 0.4, 0.5]
    upper_bound = [3.0, [1.0, 2.0], 4., 5.];
    parameter_names_new, lower_new, upper_new = parse_input(parameter_names, lower_bound, upper_bound, transform)
    @show parameter_names_new
    @show lower_new
    @show upper_new
    @test prod(parameter_names_new .== ["var", "a", "b", ["θ1", "θ2"]]) == 1
    @test prod(lower_new .== [0.3, 0.4, 0.5, 0.1, 0.2]) == 1
    @test prod(upper_new .== [3.0, 4., 5., 1, 2]) == 1
end


@testset "Test for multidimensional theta, single transformation, reorder" begin
    transform = t_sinharcsinh
    parameter_names = [ "θ", "var", "a", "b"]
    lower_bound = [[0.1, 0.2], 0.3, 0.4, 0.5]
    upper_bound = [[1.0, 2.0], 3., 4., 5.];
    parameter_names_new, lower_new, upper_new = parse_input(parameter_names, lower_bound, upper_bound, transform)
    @show parameter_names_new
    @show lower_new
    @show upper_new
    @test prod(parameter_names_new .== ["var", "a", "b", ["θ1", "θ2"]]) == 1
    @test prod(lower_new .== [0.3, 0.4, 0.5, 0.1, 0.2]) == 1
    @test prod(upper_new .== [3.0, 4., 5., 1.0, 2.0]) == 1
end
