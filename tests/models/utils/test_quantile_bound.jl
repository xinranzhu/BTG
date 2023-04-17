using Test

include("../../../src/BTG.jl")

# generate training data
ntrain = 32
noise_level = 1/1e4
randseed = 1234
trainingdata0 = get_sine_data(; ntrain=ntrain, noise_level=noise_level, 
        randseed=randseed)
ntrain = getnumpoints(trainingdata0)
xtrain = getposition(trainingdata0)
ytrain = getlabel(trainingdata0)
dimx = getdimension(trainingdata0)
dimFx = getcovariatedimension(trainingdata0)

# multi-point prediction
ntest = 101
x0 = reshape(collect(range(-pi, stop=pi, length=ntest)), ntest, 1)
Fx0 = get_sine_covariate(x0)
y0_true = get_sine_true_label(x0);

# scale to the same as xtrain and ytrain
x0 = broadcast(-, x0, trainingdata0.x_mean)
x0 = broadcast(/, x0, trainingdata0.x_std)

y0_true = broadcast(-, y0_true, trainingdata0.y_mean)
y0_true = broadcast(/, y0_true, trainingdata0.y_std)


kernel = RBFKernelModule()
t_affine = Affine(0.0, 1.0) 
t_sinharcsinh = SinhArcSinh(0.3, 1.2)
T1 = [t_affine, t_sinharcsinh]
# transform = ComposedTransformation(T1)

@testset "single transform" begin
    transform = t_sinharcsinh

    θ = Parameter("θ", 1, reshape([0.0001, 1.0], 1, 2))
    var = Parameter("var", 1, reshape([0.001, 1.0], 1, 2))
    noise_var = Parameter("noise_var", 1, reshape([0, 1e-3], 1, 2))
    params =[θ, var, noise_var] # list of Parameter
    m = Dict("θ"=>QuadInfo(θ; quad_type = QuadType(2), levels=4, num_points=2),
            "var"=>QuadInfo(var; quad_type = QuadType(2), levels=4, num_points=2),
            "noise_var"=>QuadInfo(noise_var; quad_type = QuadType(2), levels=4, num_points=2) )
    qs = QuadratureSpecs(params, m)
    qd = QuadratureDomain(qs);
    buffer = BufferDict()
    
    
        modelname="WarpedGP"
        btg0 = Btg(trainingdata0, qd, modelname, transform, kernel)

    quantile_bound = btg0.quantile_bound

    median_bound_set = Array{Float64, 2}(undef, ntest, 2)
    CI_1_bound_set = Array{Float64, 2}(undef, ntest, 2)
    CI_2_bound_set = Array{Float64, 2}(undef, ntest, 2)

    for i in 1:ntest
        #select one single testing point 
        x_i = reshape(x0[i, :], 1, dimx)
        Fx_i = reshape(Fx0[i, :], 1, dimFx)
        y_i_true = y0_true[i]
        CI_1_bound = quantile_bound(x_i, Fx_i, y_i_true*0.3, 0.025)
        median_bound = quantile_bound(x_i, Fx_i, y_i_true*0.3, 0.5)
        CI_2_bound = quantile_bound(x_i, Fx_i, y_i_true*0.3, 0.975) 
        median_bound_set[i, :] = median_bound
        CI_1_bound_set[i, :] = CI_1_bound
        CI_2_bound_set[i, :] = CI_2_bound
    end
    pred_result = btgPrediction(x0, Fx0, btg0; 
                            y_true=y0_true, confidence_level=0.95, 
                            plot_single=false, MLE=false, verbose=true);
    
    @test prod([ (pred_result.median[i] <= median_bound_set[i, 2]) && (pred_result.median[i] >= median_bound_set[i, 1]) for i in 1:ntest]) == 1
    @test prod([ (pred_result.credible_interval[i][1] <= CI_1_bound_set[i, 2]) && (pred_result.credible_interval[i][1] >= CI_1_bound_set[i, 1]) for i in 1:ntest]) == 1
    @test prod([ (pred_result.credible_interval[i][2] <= CI_2_bound_set[i, 2]) && (pred_result.credible_interval[i][2] >= CI_2_bound_set[i, 1]) for i in 1:ntest]) == 1

    # test that quantile bound does not affect MLE, run through with no error
    parameter_names = [param.name for param in qs.parameters]
    lower_bound = [0., 0., 0]
    upper_bound = [5., 5., 1.0]
    btg_optimize!(btg0, parameter_names, lower_bound, upper_bound; 
                multistart=3, randseed=1234, initial_guess=nothing, sobol=true);
    pred_result = btgPrediction(x0, Fx0, btg0; 
                y_true=y0_true, confidence_level=0.95, 
                plot_single=false, MLE=true);

end