include("../../src/BTG.jl")
include("../../src/utils/derivative/derivative_checker.jl")
using Test
# From Sept 28th notebook full workflow, Bayesian and MLE
# Tests being Bayesian with 1 quadrature node
#

@testset "full MLE, Bayesian workflow" begin
#%%
# generate training data
ntrain = 60
dat = get_nonstationary_data(;ntrain= ntrain, noise_level = 0.001)
ntrain = getnumpoints(dat)
xtrain = getposition(dat)
ytrain = getlabel(dat)
dimx = getdimension(dat)
#Plots.plot(dat.x, dat.y)

#%%
# multi-point prediction
ntest = 101
x0 = reshape(collect(range(1.0, stop=1.75, length=ntest)), ntest, 1)
y0_true= get_nonstationary_true_label(x0)
Fx0  = get_nonstationary_covariate(x0);

#%%
t1 = ArcSinh(1.0, 1.0, 1.0, 1.0)
t2 = Affine(2.0, 2.0)
t3 = SinhArcSinh(0.3, 1.2)

T1 = [t1, t2, t3]
#transform = ComposedTransformation(T1)
#transform  = ComposedTransformation([t3, t2])
transform  = ComposedTransformation([t2, t3])
#%%
θ = Parameter("θ", 1, reshape([288.10, 288.11], 1, 2))
var = Parameter("var", 1, reshape([23.598, 23.599], 1, 2))
noise_var = Parameter("noise_var", 1, reshape([4.656053,4.656054], 1, 2))
cp1 = Parameter("COMPOSED_TRANSFORM_PARAMETER_1", 2, reshape([-10.08620 2.08619; 2.08549 14.08550],2, 2))
cp2 = Parameter("COMPOSED_TRANSFORM_PARAMETER_2", 2, reshape([-2.21228 10.212281; 0.164278511 12.164278512], 2, 2))

params =[θ, var, noise_var, cp1, cp2] # list of Parameter

m = Dict("θ"=>QuadInfo(θ; quad_type = QuadType(0), levels=3, num_points=1),
        "var"=>QuadInfo(var; quad_type = QuadType(0), levels=3, num_points=1),
        "noise_var"=>QuadInfo(noise_var; quad_type = QuadType(0), levels=3, num_points=1),
        "COMPOSED_TRANSFORM_PARAMETER_1" => QuadInfo(cp1; quad_type = QuadType(0), levels = 3, num_points=1),
        "COMPOSED_TRANSFORM_PARAMETER_2" => QuadInfo(cp2; quad_type = QuadType(0), levels = 3, num_points=1))
qs = QuadratureSpecs(params, m)
qd = QuadratureDomain(qs);
buffer = BufferDict()

#%% MLE Estimate
modelname="WarpedGP"
btg2 = Btg(dat, qd, modelname, transform, kernel)

parameter_names = ["θ", "var", "noise_var", "COMPOSED_TRANSFORM_PARAMETER_1", "COMPOSED_TRANSFORM_PARAMETER_2"]
lower_bound = [0.0001, 0.00001,  0.0001, [-12.0, 0.000], [-10.0, -10.0]]
upper_bound = [300.0, 35.0, 12.0, [68.0,69.0], [49.0, 49.0]];
btg_optimize!(btg2, parameter_names, lower_bound, upper_bound;
            initial_guess=[266.21955, 23.66374, 4.451811, -3.8866676565571563, 8.0246108901450, 4.179348049227548, 6.16427851164172],
            multistart=1, randseed=32)

pred_result2 = btgPrediction(x0, Fx0, btg2;
                        y_true=y0_true, confidence_level=0.95,
                        plot_single=false, MLE=true);

nll = pred_result2.negative_log_pred_density
for i = 1:length(nll)
        @test nll[i] < 60.0
end
MSE = pred_result2.squared_error
maximum(MSE)
for i = 1:length(MSE)
        @test MSE[i]  < 50.0
end


#%% Bayesian
pred_result_bayesian = btgPrediction(x0, Fx0, btg2;
                            y_true=y0_true, confidence_level=0.95,
                            plot_single=false, MLE=false);

nll = pred_result_bayesian.negative_log_pred_density
for i = 1:length(nll)
    @test nll[i] < 60.0
end
MSE = pred_result_bayesian.squared_error
maximum(MSE)
for i = 1:length(MSE)
    @test MSE[i]  < 15.0
end
end
