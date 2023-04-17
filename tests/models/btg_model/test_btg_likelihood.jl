using Random
using Test
using ProfileView
include("../../../src/BTG.jl")
include("../../../src/utils/derivative/derivative_checker.jl")
# fix training data
ntrain = 30
noise_level = 0.1
rng = 1234
trainingdata0 = get_sine_data(; ntrain=ntrain, noise_level=noise_level, randseed=rng)
ntrain = getnumpoints(trainingdata0)
xtrain = getposition(trainingdata0)
ytrain = getlabel(trainingdata0)
Fxtrain = getcovariate(trainingdata0)
dimx = getdimension(trainingdata0)
kernel = RBFKernelModule(1.0, 2.0, 3.0)

function test()

    @testset "indentity transformation likelihood derivative test" begin
        my_transform = Id()

        cur_d = Dict("θ"=>2.5, "var"=>1.5, "noise_var"=>1.0)

        nll = BtgLikelihood(cur_d, trainingdata0, my_transform, kernel,
            BufferDict(); log_scale=true, lookup = true)

        dnll = BtgLogLikelihoodDerivatives(cur_d, trainingdata0, my_transform, kernel,
            BufferDict(); lookup = true)

        function my_f(arr)
            cur_d = Dict("noise_var"=>arr[1], "θ"=>arr[2], "var"=>arr[3])
            nll = BtgLikelihood(cur_d, trainingdata0, my_transform, kernel,
                BufferDict(); log_scale=true, lookup = true)
            return nll
        end
        my_f([1.0 ,1.0, 1.0])
        function my_df(arr)
            cur_d = Dict("noise_var"=>arr[1], "θ"=>arr[2], "var"=>arr[3])
            dnll = BtgLogLikelihoodDerivatives(cur_d, trainingdata0, my_transform, kernel,
                BufferDict(); lookup = true)
            return [dnll["noise_var"], dnll["θ"], dnll["var"]]#, dnll["var"], dnll["noise_var"]]
        end
        my_df([1.0, 1.0, 1.0])
        (r1, r2, r3, r4) = checkDerivative(my_f, my_df, 10*rand(3))
        @show r4
        @test abs(r4[1] - 2.0) < 0.2
     end
    #%%
    @testset "Single transformation likelihood derivative test" begin
        my_transform = SinhArcSinh()
        kernel = RBFKernelModule()

        cur_d = Dict("a"=>1.0, "b"=>2.0, "θ"=>2.5, "var"=>1.5, "noise_var"=>1.0)

        nll = BtgLikelihood(cur_d, trainingdata0, my_transform, kernel,
            BufferDict(); log_scale=true, lookup = true)

        dnll = BtgLogLikelihoodDerivatives(cur_d, trainingdata0, my_transform, kernel,
            BufferDict(); lookup = true)

        function my_f(arr)
            cur_d = Dict("a"=>arr[1], "b"=>arr[2], "θ"=>arr[3], "var"=>arr[4], "noise_var"=>arr[5])
            nll = BtgLikelihood(cur_d, trainingdata0, my_transform, kernel,
                BufferDict(); log_scale=true, lookup = true)
            return nll
        end
        function my_df(arr)
            cur_d = Dict("a"=>arr[1], "b"=>arr[2], "θ"=>arr[3], "var"=>arr[4], "noise_var"=>arr[5])
            dnll = BtgLogLikelihoodDerivatives(cur_d, trainingdata0, my_transform, kernel,
                BufferDict(); lookup = true)
            return [dnll["a"], dnll["b"], dnll["θ"], dnll["var"], dnll["noise_var"]]
        end
        my_f([1.0, 1.0, 1.0, 1.0, 10])
        my_df([1.0, 1.0, 1.0, 1.0, 10])

        (r1, r2, r3, r4) = checkDerivative(my_f, my_df, 10*rand(5))
        @show r4

        @test abs(r4[1] - 2.0) < 0.2
    end

    #%% test composed transform derivatives
    @testset begin "Composed transformation likelihood derivative test"
        t_affine = Affine(-0.5, 1.2)
        t_sinharcsinh = SinhArcSinh(1.0, 0.6)
        T9 = [t_affine, t_sinharcsinh]
        t_composed9 = ComposedTransformation(T9)
        cur_d = Dict("θ"=>1.3, "var"=>1.2, "noise_var"=>0.5, "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>1.0, "b"=>2.0), Dict("a"=>1.4, "b"=>1.0)])
        transform_dict =  build_input_dict(t_composed9, cur_d)

        nll = BtgLikelihood(cur_d, trainingdata0, t_composed9, kernel,
            BufferDict(); log_scale=true, lookup = true)

        dnll = BtgLogLikelihoodDerivatives(cur_d, trainingdata0, t_composed9, kernel,
            BufferDict(); lookup = true)

        parameter_names = ["θ", "var", "noise_var", [["a", "b"], ["a","b"]]]
        function my_f(arr)
            my_dict = flattened_list_to_dictionary(parameter_names, arr)
            nll = BtgLikelihood(my_dict, trainingdata0, t_composed9, kernel,
                BufferDict(); log_scale=true, lookup = true)
            return nll
        end

        function my_df(arr)
            my_dict = flattened_list_to_dictionary(parameter_names, arr)
            dnll = BtgLogLikelihoodDerivatives(my_dict, trainingdata0, t_composed9, kernel,
                BufferDict(); lookup = true)
            ret = dictionary_to_flattened_list(parameter_names, dnll)
            return ret
        end
        init = rand(7)
        my_f(init)
        my_df(init)
        (r1, r2, r3, r4) = checkDerivative(my_f, my_df, init)
        @test abs(r4[1] - 2.0) < 0.15
    end
    #%%
    all_data = RainData()
    rain1 = all_data[1]
    trainingdata0, testingdata0 = split_train_test(rain1; at=0.8,randseed=1234)
    xtrain = getposition(trainingdata0)
    ytrain = getlabel(trainingdata0)
    Fxtrain = getcovariate(trainingdata0)
    ntrain = getnumpoints(trainingdata0)
    dimx = getdimension(trainingdata0)
    dimFx = getcovariatedimension(trainingdata0)

    xtest = getposition(testingdata0)
    ytest_true = getlabel(testingdata0)
    Fxtest = getcovariate(testingdata0)
    ntest = getnumpoints(testingdata0)

    @testset "Multidimensional theta" begin
        # TODO: derivative in rbf.jl w.r.t. multi lengthscale
        t_affine = Affine(-0.5, 1.2)
        t_sinharcsinh = SinhArcSinh(1.0, 0.6)
        T9 = [t_affine, t_sinharcsinh]
        t_composed9 = ComposedTransformation(T9)
        cur_d = Dict("θ"=>[1.3, 1.2], "var"=>1.2, "noise_var"=>0.5,
                     "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>1.0, "b"=>2.0), Dict("a"=>1.4, "b"=>1.0)])
        transform_dict =  build_input_dict(t_composed9, cur_d)
        @show transform_dict

        nll = BtgLikelihood(cur_d, trainingdata0, t_composed9, kernel,
            BufferDict(); log_scale=true, lookup = true)

        dnll = BtgLogLikelihoodDerivatives(cur_d, trainingdata0, t_composed9, kernel,
            BufferDict(); lookup = true)

        parameter_names = [["θ1", "θ2"], "var", "noise_var", [["a", "b"], ["a","b"]]]
        function my_f(arr)
            my_dict = flattened_list_to_dictionary(parameter_names, arr)
            nll = BtgLikelihood(my_dict, trainingdata0, t_composed9, kernel,
                BufferDict(); log_scale=true, lookup = true)
            return nll
        end

        function my_df(arr)
            my_dict = flattened_list_to_dictionary(parameter_names, arr)
            dnll = BtgLogLikelihoodDerivatives(my_dict, trainingdata0, t_composed9, kernel,
                BufferDict(); lookup = true)
            ret = dictionary_to_flattened_list(parameter_names, dnll)
            return ret
        end

        init = rand(8)
        my_f(init)
        my_df(init)
        (r1, r2, r3, r4) = checkDerivative(my_f, my_df, init)
        @show r4
        @test abs(r4[1] - 2.0) < 0.15

    end
end

@profview test()
