using Test
include("../../src/transforms/registry.jl")
include("../../src/utils/derivative/derivative_checker.jl")

## WARNING: DO NOT TEST DERIVATIVE WITH AFFINE AS LAST ELEMENT IN LIST
# AFFINE DOES NOT PLAY WELL WITH FINITE DIFFERENCE CHECKER

# initialize several elementary transforms
t_affine = Affine(-0.5, 1.2) # f(y) = -0.5 + 1.2y
t_boxcox = BoxCox(0.8)
t_arcsinh = ArcSinh(1.0, 0.6, 0.2, 2.2)
t_sinharcsinh = SinhArcSinh(0.45, 1.55)

T1 = [t_arcsinh, t_affine, t_sinharcsinh]
T2 = [t_affine, t_arcsinh]
T3 = [t_affine]
T5 = [t_arcsinh]
T6 = [t_sinharcsinh]
T7 = [t_arcsinh, t_sinharcsinh]
T8 = [t_sinharcsinh]
T9 = [t_affine, t_sinharcsinh]
T10 = [t_sinharcsinh, t_boxcox]
T11 = [t_affine, t_boxcox]
T12 = [t_arcsinh, t_boxcox]
T13 = [t_affine, t_sinharcsinh, t_boxcox]

t_composed1 = ComposedTransformation(T1)
t_composed2 = ComposedTransformation(T2)
t_composed3 = ComposedTransformation(T3)
t_composed5 = ComposedTransformation(T5)
t_composed6 = ComposedTransformation(T6)
t_composed7 = ComposedTransformation(T7)
t_composed8 = ComposedTransformation(T8)
t_composed9 = ComposedTransformation(T9)
t_composed9 = ComposedTransformation(T9)
t_composed10 = ComposedTransformation(T10)
t_composed11 = ComposedTransformation(T11)
t_composed12 = ComposedTransformation(T12)
t_composed13 = ComposedTransformation(T13)


#%%
# cur_d = Dict("COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>1.0), Dict("b"=>1.0)])
# transform_dict =  build_input_dict(t_composed9, cur_d)


# g(y) = evaluate(t_composed9, transform_dict, y[1])
# #evaluate_derivative_hyperparameters(transform, transform_dict, y)
# #evalute_derivative_x_hyperparameters(transform, transform_dict, y)
# jac(y) = evaluate_derivative_x(t_composed9, transform_dict, y[1])
# g(1.3)
# jac(1.3)
# (r1, r2, r3, r4) = checkDerivative(g, jac, [1.2])
# @show r4


# #%%
# @testset "basic evaluation" begin
#     aa = evaluate(t_arcsinh, 1.0)
#     bb = evaluate(t_affine, aa)
#     cc = evaluate(t_composed2, [Dict("a"=>-0.5, "b"=>1.2), Dict("a"=>1.0, "b"=>0.6, "c"=>0.2, "d"=>2.2)], 1.0)
#     @test cc == bb
# end

# #%%
# @testset "derivative of composed transform w.r.t y" begin
#     function evaluate_y(y)
#         T1 = [t_arcsinh, t_sinharcsinh, t_affine]
#         array_dict = Array{Dict}(undef, 3)
#         for i in 1:3
#             array_dict[i] = build_input_dict(T1[i], Dict())
#         end
#         return evaluate(t_composed1, array_dict, y[1])
#     end

#     function derivative_y(y)
#         T1 = [t_arcsinh, t_sinharcsinh, t_affine]
#         array_dict = Array{Dict}(undef, 3)
#         for i in 1:3
#             array_dict[i] = build_input_dict(T1[i], Dict())
#         end
#         return evaluate_derivative_x(t_composed1, array_dict, y[1])
#     end

#     (r1, r2, r3, r4) = checkDerivative(evaluate_y, derivative_y, [1.1])
#     @test abs(r4[1]- 2.00) < 1e-1
# end

# #%%

# function evaluate(arr)
#     T = [t_arcsinh, t_sinharcsinh, t_affine]
# #     T = [t_arcsinh]
#     t_composed = ComposedTransformation(T)
#     param_length = [1, 5, 7]
#     n_transforms = length(T)
#     array_dict = Array{Dict}(undef, n_transforms)
#     for i = 1:n_transforms
#         names = T[i].names
#         n_params = length(names)
#         dict= Dict()
#         for j in 1:n_params
#             dict[names[j]] = arr[param_length[i]+j-1]
#         end
#         array_dict[i] = dict
#     end
#     return evaluate(t_composed, array_dict, 1.0)
# end

# function derivative_param(arr)
#     T = [t_arcsinh, t_sinharcsinh, t_affine]
# #     T = [t_arcsinh]
#     param_length = [1, 5, 7]
#     t_composed = ComposedTransformation(T)
#     n_transforms = length(T)
#     array_dict = Array{Dict}(undef, n_transforms)
#     for i in 1:n_transforms
#         names = T[i].names
#         n_params = length(names)
#         dict= Dict()
#         for j in 1:n_params
#             dict[names[j]] = arr[param_length[i]+j-1]
#         end
#         array_dict[i] = dict
#     end
#     res = evaluate_derivative_hyperparams(t_composed, array_dict, 1.0)
#     out = []
#     for i = 1:n_transforms
#         names = T[i].names
#         dict_deriv = res[i]
#         for name in names
#             out = cat(out, dict_deriv[name], dims=1)
#         end
#     end
#     return out
# end
# (r1, r2, r3, r4) = checkDerivative(evaluate, derivative_param, 3*rand(8));

# @testset "derivative of composed transform w.r.t hyperparameters" begin
#     @test abs(r4[1] - 2.0) < 0.1
# end


# #%%
# @testset "derivative of composed transform w.r.t y and hyperparameters" begin
#     function evaluate2(arr)
#         T = [t_arcsinh, t_affine, t_sinharcsinh]
#     #     T = [t_arcsinh]
#         t_composed = ComposedTransformation(T)
#         param_length = [1, 5, 7]
#         n_transforms = length(T)
#         array_dict = Array{Dict}(undef, n_transforms)
#         for i = 1:n_transforms
#             names = T[i].names
#             n_params = length(names)
#             dict= Dict()
#             for j in 1:n_params
#                 dict[names[j]] = arr[param_length[i]+j-1]
#             end
#             array_dict[i] = dict
#         end
#         return evaluate_derivative_x(t_composed, array_dict, 1.545)
#     end

#     function derivative_param2(arr)
#         T = [t_arcsinh, t_affine, t_sinharcsinh]
#     #     T = [t_arcsinh]
#         param_length = [1, 5, 7]
#         t_composed = ComposedTransformation(T)
#         n_transforms = length(T)
#         array_dict = Array{Dict}(undef, n_transforms)
#         for i in 1:n_transforms
#             names = T[i].names
#             n_params = length(names)
#             dict= Dict()
#             for j in 1:n_params
#                 dict[names[j]] = arr[param_length[i]+j-1]
#             end
#             array_dict[i] = dict
#         end
#         res = evaluate_derivative_x_hyperparams(t_composed, array_dict, 1.545)
#         #@show res
#         out = []
#         for i = 1:n_transforms
#             names = T[i].names
#             dict_deriv = res[i]
#             for name in names
#                 out = cat(out, dict_deriv[name], dims=1)
#             end
#         end
#         return out
#     end
#     (r1, r2, r3, r4) = checkDerivative(evaluate2, derivative_param2, 10*rand(8))
#     @test abs(r4[1] - 2.0)<1e-1
# end

# @testset "Test build input dict" begin
#     # imagine the d comes from reformat(node_dict), where node_dict is from iterator
#     # XZ: so reformat should at least give list of empty Dict: "COMPOSED_TRANSFORM_DICTS"=>[Dict(), Dict()]
#     d = Dict("noise_var" => 2., "θ" => 0.03465,
#                     "var" => 0.66645, 
#                     "COMPOSED_TRANSFORM_DICTS"=>[Dict(), Dict()])
#     d_t2_complete = build_input_dict(t_composed2, d)
#     @test length(d_t2_complete) == 2
#     @test d_t2_complete[1] == t_composed2.transform_list[1].PARAMETER_DICT
#     @test d_t2_complete[2] == t_composed2.transform_list[2].PARAMETER_DICT

#     d = Dict("noise_var" => 2., "θ" => 0.03465,
#                     "var" => 0.66645, 
#                     "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>999.), Dict()])
#     d_t2_complete = build_input_dict(t_composed2, d)
#     @test length(d_t2_complete) == 2
#     @test d_t2_complete[1]["b"] == t_composed2.transform_list[1].PARAMETER_DICT["b"]
#     @test d_t2_complete[1]["a"] == 999.
#     @test d_t2_complete[2] == t_composed2.transform_list[2].PARAMETER_DICT
# end


# @testset "Test inverse" begin
    x = rand()
    array_dict_i = [Dict(), Dict()]
    t_composed_i = t_composed9
    y = evaluate_inverse(t_composed_i, array_dict_i, x)
    x_text = evaluate(t_composed_i, array_dict_i, y)
    @test abs(x-x_text) < 1e-3

    x = rand()
    array_dict_i = [Dict(), Dict()]
    t_composed_i = t_composed10
    y = evaluate_inverse(t_composed_i, array_dict_i, x)
    x_text = evaluate(t_composed_i, array_dict_i, y)
    @test abs(x-x_text) < 1e-3

    x = rand()
    array_dict_i = [Dict(), Dict()]
    t_composed_i = t_composed11
    y = evaluate_inverse(t_composed_i, array_dict_i, x)
    x_text = evaluate(t_composed_i, array_dict_i, y)
    @test abs(x-x_text) < 1e-3

    x = rand()
    array_dict_i = [Dict(), Dict()]
    t_composed_i = t_composed12
    y = evaluate_inverse(t_composed_i, array_dict_i, x)
    x_text = evaluate(t_composed_i, array_dict_i, y)
    @test abs(x-x_text) < 1e-3


    x = rand()
    array_dict_i = [Dict(), Dict(), Dict()]
    t_composed_i = t_composed13
    y = evaluate_inverse(t_composed_i, array_dict_i, x)
    x_text = evaluate(t_composed_i, array_dict_i, y)
    @test abs(x-x_text) < 1e-3





# end