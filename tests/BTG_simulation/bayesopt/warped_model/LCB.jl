include("../../../../src/BTG.jl")
using Test
#%%
@testset "LCB acquisition function multidimensional x" begin
mykernel = RBFKernelModule([1.0, 5.0], 2.0, 3.0)
mytransform  = SinhArcSinh(1.0, 1.0)
mydomain = getQuadratureDomain(;θ_dim=2)
mytd = get_random_training_data(;d=2, seed = 22)
mybtg = Btg(mytd, mydomain, "WarpedGP", mytransform, mykernel; lookup=true);
parameter_names = ["a", "b", "θ", "var", "noise_var"]
lower_bound = [-20., 0., [0.0, 0.0],   0,   0]
upper_bound = [25., 35.0, [150.0, 200.0], 45.0, 32.5];
btg_optimize!(mybtg, parameter_names, lower_bound, upper_bound;
                multistart=5, randseed=1234, initial_guess=nothing, sobol=false)
mybtg.likelihood_optimum
getposition(mytd)
h(x) = LCB_untransformed(reshape(x, 1, length(x)), mybtg)
dh(x) = LCB_untransformed_derivative(reshape(x, 1, length(x)), mybtg)
w(x) = LCB(reshape(x, 1, length(x)), mybtg)
dw(x) = LCB_derivative(reshape(x, 1, length(x)), mybtg)
z = reshape([0.8, 0.4], 1, 2)
w(z)
dw(z)
(_, _, _, r4) = checkDerivative(w, dw, z)
#@show r4
@test abs(r4[1]-2)<0.5
z = [.5, .5]
h(z)
dh(z)
(_, _, _, r4) = checkDerivative(h, dh, z)
#@show r4
@test abs(r4[1]-2)<0.5
end
#%%
@testset "LCB acquisition function single dimensional x" begin
mykernel = RBFKernelModule(1.2, 2.0, 3.0)
mytransform  = SinhArcSinh(1.0, 1.0)
mydomain = getQuadratureDomain(;θ_dim=1) #dimension of theta must equal dimension of x
mytd = get_random_training_data(;d=1, seed = 22)
mybtg = Btg(mytd, mydomain, "WarpedGP", mytransform, mykernel; lookup=true);
parameter_names = ["a", "b", "θ", "var", "noise_var"]
lower_bound = [-20., 0., 0.0,   0,   0]
upper_bound = [25., 35.0, 200.0, 45.0, 32.5];
btg_optimize!(mybtg, parameter_names, lower_bound, upper_bound;
                multistart=5, randseed=1234, initial_guess=nothing, sobol=false)
mybtg.likelihood_optimum
getposition(mytd)
h(x) = LCB_untransformed(reshape(x, 1, length(x)), mybtg; β = 2.0)
dh(x) = LCB_untransformed_derivative(reshape(x, 1, length(x)), mybtg; β = 2.0)
dh_alternate(x) = (store = [1.0]; LCB_untransformed_derivative!(store, reshape(x, 1, length(x)), mybtg; β = 2.0); store)
w(x) = LCB(reshape(x, 1, length(x)), mybtg)
dw(x) = LCB_derivative(reshape(x, 1, length(x)), mybtg)
dw_alternate(x) = (store = [1.0]; LCB_derivative!(store, reshape(x, 1, length(x)), mybtg; β = 2.0); store)

z = reshape([0.35], 1, 1)

w(z)
dw(z)
(_, _, pl, r4) = checkDerivative(w, dw, z, nothing, 3, 10)
@show r4
@test r4[1]-2 > -0.5
(_, _, pl, r4) = checkDerivative(w, dw_alternate, z, nothing, 3, 10)
@show r4
dw_alternate([.35])
dw([.35])
@test r4[1]-2 > -0.5

(_, _, _, r4) = checkDerivative(h, dh, z, nothing, 1, 4)
@show r4
@test r4[1]-2 > -0.5
(_, _, _, r4) = checkDerivative(h, dh_alternate, z, nothing, 1, 4)
#@show r4
@test r4[1]-2 > -0.5
end
