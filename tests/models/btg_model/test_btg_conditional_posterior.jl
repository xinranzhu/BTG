
using Test

include("../../../src/BTG.jl")

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

ntest = 1
x0 = reshape([0.222], ntest, 1)
Fx0 = get_sine_covariate(x0)
y0_true = get_sine_true_label(x0)
y_mean = trainingdata0.y_mean
y_std = trainingdata0.y_std
y0_true = (y0_true - y_mean ) / y_std

# Test 1: single transform, transform params only
@testset "Test 1: single trans, trans params only" begin
    param_dict_fixed = Dict("θ"=> 1.1, "var"=>1.0, "noise_var"=>0.001)
    transform = SinhArcSinh()
    kernel = RBFKernelModule(param_dict_fixed)
    dict = Dict("a"=>0.2, "b"=>1.2)
    y0 = 0.99*y0_true
    posterior_pdf, posterior_cdf, quant_val = BtgConditionalPosterior(dict, x0, Fx0, y0, trainingdata0, transform, kernel, BufferDict();
        lookup = true, test_buffer = Buffer(), quant=0.5)

    @test posterior_pdf > 0
    @test posterior_cdf > 0
    @test quant_val > 0
end

# Test 2: single transform, kernel params only
@testset "Test 2: single trans, kernel params only" begin
    param_dict_fixed = Dict("a"=>0.2, "b"=>1.2)
    transform = SinhArcSinh(param_dict_fixed)
    kernel = RBFKernelModule()
    dict = Dict(Dict("θ"=> 1.1, "var"=>1.0, "noise_var"=>0.001))
    y0 = 0.99*y0_true
    posterior_pdf, posterior_cdf, quant_val = BtgConditionalPosterior(dict, x0, Fx0, y0, trainingdata0, transform, kernel, BufferDict();
        lookup = true, test_buffer = Buffer(), quant=0.5)

    @test posterior_pdf > 0
    @test posterior_cdf > 0
    @test quant_val > 0
end


# Test 3: single transform, all params
@testset "Test 3: single trans, all params" begin
    transform = SinhArcSinh()
    kernel = RBFKernelModule()
    dict = Dict("θ"=> 1.1, "var"=>1.0, "noise_var"=>0.001, "a"=>0.2, "b"=>1.2)
    y0 = 0.99*y0_true
    posterior_pdf, posterior_cdf, quant_val = BtgConditionalPosterior(dict, x0, Fx0, y0, trainingdata0, transform, kernel, BufferDict();
        lookup = true, test_buffer = Buffer(), quant=0.5)

    @test posterior_pdf > 0
    @test posterior_cdf > 0
    @test quant_val > 0
end

# Test 4: composed transform, all params
@testset "Test 4: composed trans, all params" begin

    t_arcsinh = ArcSinh()
    t_affine = Affine()
    t_sinharcsinh = SinhArcSinh()
    T1 = [t_arcsinh, t_affine, t_sinharcsinh]
    transform = ComposedTransformation(T1)
    kernel = RBFKernelModule()
    dict = Dict("COMPOSED_TRANSFORM_PARAMETER_1"=>[1.0, 0.6, 0.2, 2.2],
                "COMPOSED_TRANSFORM_PARAMETER_2"=>[-0.5, 1.2],
                "COMPOSED_TRANSFORM_PARAMETER_3"=>[0.45, 1.55],
                "θ"=> 1.1, "var"=>1.0, "noise_var"=>0.001)
    y0 = 0.99*y0_true

    posterior_pdf, posterior_cdf, quant_val = BtgConditionalPosterior(dict, x0, Fx0, y0, trainingdata0, transform, kernel, BufferDict();
        lookup = true, test_buffer = Buffer(), quant=0.5)

    @test posterior_pdf > 0
    @test posterior_cdf > 0
    @test quant_val > 0
end
