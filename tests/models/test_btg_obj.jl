using Random
using Test

include("../../src/BTG.jl")

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

# Test1: be bayesian about both transform params and kernel params
#@testset "Test BTG model fields" begin

my_transform = SinhArcSinh()
param_dict_fixed = Dict("var"=>1.0, "noise_var"=>0.001)
kernel = RBFKernelModule(param_dict_fixed)

a = Parameter("a", 1, reshape([0, 5.0], 1, 2))
b = Parameter("b", 1, reshape([1.0, 6.0], 1, 2))
θ = Parameter("θ", 1, reshape([0.1, 1.0], 1, 2))
params =[a, b, θ] # list of Parameter
params_names = [param.name for param in params]
m = Dict("a"=>QuadInfo(a; quad_type =  QuadType(3), levels=2, num_points=4),
        "b"=>QuadInfo(b; quad_type =  QuadType(3), levels=2, num_points=4),
        "θ"=>QuadInfo(θ; quad_type =  QuadType(3), levels=2, num_points=5))
qs = QuadratureSpecs(params, m)
qd = QuadratureDomain(qs)

modelname="WarpedGP"
btg0 = Btg(trainingdata0, qd, modelname, my_transform, kernel)

#test each field in btg0
#trainingdata
trainingdata_test = btg0.trainingdata
ntrain_test = getnumpoints(trainingdata_test)
xtrain_test = getposition(trainingdata_test)
ytrain_test = getlabel(trainingdata_test)
Fxtrain_test = getcovariate(trainingdata_test)
dimx_test = getdimension(trainingdata_test)

@test ntrain == ntrain_test
@test prod(xtrain_test .== xtrain) == 1
@test prod(ytrain_test .== ytrain) == 1
@test prod(Fxtrain_test .== Fxtrain) == 1
@test dimx_test == dimx

#kernel
kernel_test = btg0.kernel
@test typeof(kernel_test)<:RBFKernelModule #type checking
kernel_param = kernel_test.PARAMETER_DICT # parameter checking
#XZ: do we want default values in rbf definition or nothing?
@test kernel_param["θ"] == nothing
@test kernel_param["noise_var"] == 0.001
@test kernel_param["var"] == 1.0

#transform
transform_test = btg0.transform
@test typeof(transform_test)<:SinhArcSinh
transform_dict = transform_test.PARAMETER_DICT
@test transform_dict["a"] == nothing
@test transform_dict["b"] == nothing

#domain
qd_test = btg0.domain
@test typeof(qd_test)<:QuadratureDomain

#buffer, type checkingbuffer_test = btg0.buffer
buffer_test = btg0.buffer
@test typeof(buffer_test)<:BufferDict

#lookup, check status
lookup_test = btg0.lookup
@test btg0.lookup

#iter_normalized, check it's working and weights sum to 1
iter_normalized_test = btg0.iter_normalized
@test typeof(iter_normalized_test)<:Base.Generator
@test sum([w for (n, w) in iter_normalized_test]) ≈ 1.0 rtol = 0.2
for (n, w) in iter_normalized_test
    # assert key in quadrature nodes are
    @assert ( issubset(keys(n),params_names) ) && ( issubset(params_names, keys(n)) )
end

# MLE_optimize_status, check it's nothing
@test btg0.MLE_optimize_status == nothing

#likelihood_optimum, check empty Dict
likelihood_optimum_test = btg0.likelihood_optimum
@test length(keys(likelihood_optimum_test.first)) == 0
@test likelihood_optimum_test.second == 0

ntest = 1
x0 = reshape([1.0], ntest, 1)
Fx0 = get_sine_covariate(x0)
y0_true = get_sine_true_label(x0)[1]

#conditional_posterior, check evaluation
conditional_posterior_test(d, x0, Fx0, y0) = btg0.conditional_posterior(d, x0, Fx0, y0,
                                        trainingdata_test, transform_test, kernel_test,
                                        buffer_test; lookup=lookup_test)
d = Dict("θ"=>0.5, "a"=>0.0, "b"=>1.0)
@test prod(conditional_posterior_test(d, x0, Fx0, y0_true)[1:2] .>= [0., 0.])

#likelihood, check evaluation
likelihood_test = btg0.likelihood
likelihood_test_fixed(d) = likelihood_test(d, trainingdata_test, transform_test,
                                            kernel_test, buffer_test;
                                            log_scale=false, lookup=lookup_test)
d = Dict("θ"=>0.2, "a"=>0.1, "b"=>0.8)
@test likelihood_test_fixed(d) >= 0. # test evaluation
d = Dict("θ"=>0.3, "a"=>0.0, "b"=>1.0)
@test likelihood_test_fixed(d) >= 0. # test evaluation

#posterior_pdf, check evaluation and integral
posterior_pdf_test = btg0.posterior_pdf
posterior_pdf_test_fixed(y0) = posterior_pdf_test(x0, Fx0, y0)
posterior_pdf_test_fixed(y0_true)
@test posterior_pdf_test_fixed(y0_true) >= 0. # test evaluation
res = hquadrature(posterior_pdf_test_fixed, 0., 1.1) # res = [integral value, error bound]
@test abs(res[1] - 1) <= res[2] # assert int_PDF = 1

#posterior_cdf, check evaluation
posterior_cdf_test = btg0.posterior_cdf
posterior_cdf_test_fixed(y0) = posterior_cdf_test(x0, Fx0, y0)
@test posterior_cdf_test_fixed(y0_true) >= 0. # test evaluation
#@test posterior_cdf_test_fixed(1.0) ≈ 1.0 rtol = 0.1 # assert CDF(1) = 1



#end
#%%
# Test2: be bayesian about kernel params only, trivial transform (identity)
#@testset "Test BTG model fields" begin

my_transform = Id()
param_dict_fixed = Dict("var"=>1.0, "noise_var"=>0.001)
kernel = RBFKernelModule(param_dict_fixed)


θ = Parameter("θ", 1, reshape([0.1, 1.0], 1, 2))
params =[θ] # list of Parameter
params_names = [param.name for param in params]
m = Dict("θ"=>QuadInfo(θ; quad_type = QuadType(3), levels=5, num_points=5))
qs = QuadratureSpecs(params, m)
qd = QuadratureDomain(qs)

modelname="WarpedGP"
btg0 = Btg(trainingdata0, qd, modelname, my_transform, kernel)

##test each field in btg0
#trainingdata
trainingdata_test = btg0.trainingdata
ntrain_test = getnumpoints(trainingdata_test)
xtrain_test = getposition(trainingdata_test)
ytrain_test = getlabel(trainingdata_test)
Fxtrain_test = getcovariate(trainingdata_test)
dimx_test = getdimension(trainingdata_test)
@test ntrain == ntrain_test
@test prod(xtrain_test .== xtrain) == 1
@test prod(ytrain_test .== ytrain) == 1
@test prod(Fxtrain_test .== Fxtrain) == 1
@test dimx_test == dimx

#kernel
kernel_test = btg0.kernel
@test typeof(kernel_test)<:RBFKernelModule #type checking
kernel_param = kernel_test.PARAMETER_DICT # parameter checking
#XZ: do we want default values in rbf definition or nothing?
@test kernel_param["θ"] == nothing
@test kernel_param["noise_var"] == 0.001
@test kernel_param["var"] == 1.0

#transform
transform_test = btg0.transform
@test typeof(transform_test)<:Id
@test transform_test.PARAMETER_DICT == Dict()


#domain
qd_test = btg0.domain
@test typeof(qd_test)<:QuadratureDomain

#buffer, type checkingbuffer_test = btg0.buffer
buffer_test = btg0.buffer
@test typeof(buffer_test)<:BufferDict

#lookup, check status
lookup_test = btg0.lookup
@test btg0.lookup

#iter_normalized, check it's working and weights sum to 1
iter_normalized_test = btg0.iter_normalized
@test typeof(iter_normalized_test)<:Base.Generator
@test sum([w for (n, w) in iter_normalized_test]) ≈ 1.0 rtol = 0.02
for (n, w) in iter_normalized_test
    # assert key in quadrature nodes are
    @assert ( issubset(keys(n),params_names) ) && ( issubset(params_names, keys(n)) )
end

#MLE_optimize_status, check it's nothing
@test btg0.MLE_optimize_status == nothing

#likelihood_optimum, check empty Dict
likelihood_optimum_test = btg0.likelihood_optimum
@test length(keys(likelihood_optimum_test.first)) == 0
@test likelihood_optimum_test.second == 0

ntest = 1
x0 = reshape([1.0], ntest, 1)
Fx0 = get_sine_covariate(x0)
y0_true = get_sine_true_label(x0)[1]

#conditional_posterior, check evaluation
conditional_posterior_test(d, x0, Fx0, y0) = btg0.conditional_posterior(d, x0, Fx0, y0,
                                        trainingdata_test, transform_test, kernel_test,
                                        buffer_test; lookup=lookup_test)
d = Dict("θ"=>0.5)
@test prod(conditional_posterior_test(d, x0, Fx0, y0_true)[1:2] .>= [0., 0.])

#likelihood, check evaluation
likelihood_test = btg0.likelihood
likelihood_test_fixed(d) = likelihood_test(d, trainingdata_test, transform_test,
                                            kernel_test, buffer_test;
                                            log_scale=false, lookup=lookup_test)
d = Dict("θ"=>0.2)
@test likelihood_test_fixed(d) >= 0. # test evaluation
d = Dict("θ"=>0.3)
@test likelihood_test_fixed(d) >= 0. # test evaluation

#posterior_pdf, check evaluation and integral
posterior_pdf_test = btg0.posterior_pdf
posterior_pdf_test_fixed(y0) = posterior_pdf_test(x0, Fx0, y0)
@test posterior_pdf_test_fixed(y0_true) >= 0. # test evaluation
int = hquadrature(posterior_pdf_test_fixed, 0., 1.1)
@test abs(int[1] - 1) <= 0.01 # assert int_PDF = 1

#posterior_cdf, check evaluation
posterior_cdf_test = btg0.posterior_cdf
posterior_cdf_test_fixed(y0) = posterior_cdf_test(x0, Fx0, y0)
@test posterior_cdf_test_fixed(y0_true) >= 0. # test evaluation
#@show posterior_cdf_test_fixed(1.0)
#@test posterior_cdf_test_fixed(1.0) ≈ 1.0 rtol = 0.1 # assert CDF(1) = 1

#end


# @testset "Test btg object for 2d data" begin

    # all_data = RainData()
    # rain1 = all_data[1]
    # trainingdata0, testingdata0 = split_train_test(rain1, at=0.9)
    # xtrain = getposition(trainingdata0)
    # ytrain = getlabel(trainingdata0)
    # dimx = getdimension(trainingdata0)
    # ntrain = getnumpoints(trainingdata0)
    # xtest = getposition(testingdata0)
    # Fxtest = getcovariate(testingdata0)
    # ytest_true = getlabel(testingdata0);

    # kernel = RBFKernelModule()
    # transform = SinhArcSinh()

    # a = Parameter("a", 1, reshape([0, 1.0], 1, 2))
    # b = Parameter("b", 1, reshape([0., 1.0], 1, 2))
    # θ = Parameter("θ", 2, reshape([0 1.0; 0 1], 2, 2))
    # var = Parameter("var", 1, reshape([0., 1.], 1, 2))
    # noise_var = Parameter("noise_var", 1, reshape([0, 1.0], 1, 2))
    # params =[a, b, θ, var, noise_var] # list of Parameter
    # m = Dict("a"=>QuadInfo(a; quad_type = QuadType(3), levels=4, num_points=4),
    #         "b"=>QuadInfo(b; quad_type = QuadType(3), levels=4, num_points=4),
    #         "θ"=>QuadInfo(θ; quad_type = QuadType(3), levels=4, num_points=5),
    #         "var"=>QuadInfo(var; quad_type = QuadType(3), levels=4, num_points=5),
    #         "noise_var"=>QuadInfo(noise_var; quad_type = QuadType(3), levels=4, num_points=5))
    # qs = QuadratureSpecs(params, m)
    # qd = QuadratureDomain(qs);

    # btg0 = Btg(trainingdata0, qd, conditional_posterior, likelihood,
    #         transform, kernel; lookup=true);





# end
