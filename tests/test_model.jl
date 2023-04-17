include("../src/BTG.jl")


# generate training data
ntrain = 32
noise_level = 1/1e4
randseed = 1234
trainingdata0 = get_sine_data(; ntrain=ntrain, noise_level=noise_level, randseed=randseed)
ntrain = getnumpoints(trainingdata0)
xtrain = getposition(trainingdata0)
ytrain = getlabel(trainingdata0)
dimx = getdimension(trainingdata0)

# multi-point prediction
ntest = 101
x0 = reshape(collect(range(-pi, stop=pi, length=ntest)), ntest, 1)
Fx0 = get_sine_covariate(x0)
y0_true = get_sine_true_label(x0)

# Model via GP pkg -- trained on the normalized data! 
modelname = "GP"
mymean = MeanZero() 
kern = SE(zeros(dimx), 0.0) 
xtrain_GP = reshape(xtrain, dimx, ntrain)
ytrain_GP = ytrain
gpmodel = GP(xtrain_GP, ytrain_GP, mymean, kern) 
GaussianProcesses.optimize!(gpmodel)
GPResults = GpPrediction(trainingdata0, x0, Fx0, gpmodel; y_true = y0_true)

# show_results(GPResults; verbose=false, plot=false, normalization=true)



kernel = RBFKernelModule()
transform = Id()
θ = Parameter("θ", 1, reshape([0.0346, 0.0347], 1, 2)) # range doesn't matter
var = Parameter("var", 1, reshape([0.6664, 0.6665], 1, 2)) # range doesn't matter
noise_var = Parameter("noise_var", 1, reshape([2.94e-5, 2.95e-5], 1, 2)) # range doesn't matter
params =[θ, var, noise_var] # list of Parameter
m = Dict("θ"=>QuadInfo(θ; quad_type = QuadType(3), levels=4, num_points=5),
        "var"=>QuadInfo(var; quad_type = QuadType(3), levels=4, num_points=5),
        "noise_var"=>QuadInfo(var; quad_type = QuadType(3), levels=4, num_points=5))
optimal_guess = nothing
# optimal_guess = [0.03468690191664504, 0.6664197424659871, 2.94862212769642e-5];


qs = QuadratureSpecs(params, m)
qd = QuadratureDomain(qs)
parameter_names = [param.name for param in qs.parameters]

modelname="WarpedGP"
btg0 = Btg(trainingdata0, qd, modelname, transform, kernel)


# get the optimal parameter values
likelihood_optimum = GPResults.likelihood_optimum
optimal_guess = []
for name in parameter_names
    push!(optimal_guess, (likelihood_optimum.first[name])[1])
end

lower_bound = [0.00001, 0.00001, 0.000001]
upper_bound = [1.5, 1.0, 0.1]
btg_optimize!(btg0, parameter_names, lower_bound, upper_bound; 
              multistart=100, randseed=1234, initial_guess=optimal_guess)

pred_result = btgPrediction(x0, Fx0, btg0; 
                            y_true=y0_true, confidence_level=0.95, 
                            plot_single=false, MLE=MLE)

# show_results(pred_result; verbose=false)

println("----Compare model parameter----")
@show GPResults.likelihood_optimum
@show pred_result.likelihood_optimum

@test abs(mean(GPResults.squared_error) - mean(pred_result.squared_error)) < 1e-3
@test abs(mean(GPResults.absolute_error) - mean(pred_result.absolute_error)) < 1e-3
@test abs(mean(GPResults.negative_log_pred_density) - mean(pred_result.negative_log_pred_density)) < 1e-3


    

