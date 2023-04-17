include("../src/BTG.jl")
ntrain = 10
trainingdata0 = get_int_sine_data(; ntrain=ntrain)
xtrain = getposition(trainingdata0)
ytrain = getlabel(trainingdata0)
dimx = getdimension(trainingdata0)
feature_size = 1
@assert dimx == feature_size

# multi-point prediction
ntest = 400
x0 = reshape(collect(range(-pi, stop=pi, length=ntest)), ntest, 1)
Fx0 = get_int_sine_covariate(x0)
y0_true = get_int_sine_true_label(x0);

transform_name = "ArcSinh"
modelname = "Btg"

model, param_info, transform_list = init_btg(trainingdata0, transform_name);
btg_model = fit_btg(model, trainingdata0, param_info, transform_list;
    total_mass=0.98);