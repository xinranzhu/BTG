#multiple length scales
include("kernel_module.jl")
include("Utils/utils.jl")
include("../utils/derivative/derivative_checker.jl")
using Distances
import Distances.pairwise
import .utils.reshape
#%% Define KernelModuleType and KernelModule struct
#struct RBFKernelModuleType <: KernelModuleType
#    function RBFKernelModuleType()
#        return new()
#    end
#end
mutable struct RBFKernelModule <:KernelModule
     θ::Union{Array{Float64}, Float64, Nothing}
     var::Union{Array{Float64}, Float64, Nothing} #signal variance
     noise_var::Union{Array{Float64}, Float64, Nothing} #noise variance
     PARAMETER_DICT::Dict
     names::Array{W} where W<:String
     function RBFKernelModule(θ, var, noise_var)
         return new(θ, var, noise_var, Dict("θ"=>θ, "var"=>var, "noise_var"=>noise_var), ["θ", "var", "noise_var"])
     end
     function RBFKernelModule(dict::Dict)
         names = ["θ", "var", "noise_var"]
         if !issubset(Set(keys(dict)), Set(names))
            error("initializer dict d must have keys in {\"θ\", \"var\", \"noise_var\"}")
         end
         θ = get(dict, "θ", nothing)
         var = get(dict, "var", nothing)
         noise_var = get(dict, "noise_var", nothing)
         dict_new = Dict("θ"=>θ, "var"=>var, "noise_var"=>noise_var)
         return new(θ, var, noise_var, dict_new, names)
     end
     function RBFKernelModule()
         return new(nothing, nothing, nothing, Dict("θ"=>nothing, "var"=>nothing, "noise_var"=>nothing),  ["θ", "var", "noise_var"])
     end
end

"""
Core functionality of forming kernel matrix is accessed through kernel matrix.jl
"""
function evaluate(kernel::RBFKernelModule, a::Array{T}, b::Array{T};
    θ = kernel.θ, var = kernel.var, noise_var = kernel.noise_var, jitter = 1e-10, dims = 1) where T<:Real
    if θ == nothing || typeof(θ)<:Real
        metric = SqEuclidean()
        distances = pairwise(metric, reshape(a), reshape(b), dims=1)
        ret = var[1] * exp.(-0.5 * θ * distances)
        ret[LinearAlgebra.diagind(ret)] .+= jitter
        ret[LinearAlgebra.diagind(ret)] .+= noise_var[1] #account for noise
        return ret
    elseif typeof(θ)<:Array{T} where T<:Real
        metric = WeightedSqEuclidean(reshape(θ))
        distances = pairwise(metric, reshape(a), reshape(b), dims=1)
        ret =  var[1] * exp.(-0.5 * distances)
        ret[LinearAlgebra.diagind(ret)] .+= jitter
        ret[LinearAlgebra.diagind(ret)] .+= noise_var[1] #account for noise
        return ret
    else
        error("failure in evaluate kernel")
    end
end


function evaluate_derivative_θ(kernel::RBFKernelModule, a::Array{T}, b::Array{T};
    θ = kernel.θ, var = kernel.var, noise_var = kernel.noise_var, jitter = 1e-10, dims = 1) where T<:Real
    if length(θ) == 1 || typeof(θ)<:Real
        θ = θ[1]
        metric = SqEuclidean()
        distances = pairwise(metric, reshape(a), reshape(b), dims=1)
        ret = var[1] * exp.(-0.5 * θ * distances) .* distances .* (-0.5)
        ret[LinearAlgebra.diagind(ret)] .+= jitter
        return ret
    elseif typeof(θ)<:Array{T} where T<:Real
        # error("only derivative w.r.t single variable θ is supported")
        ret_set = Any[]
        Kernel_mat = evaluate(kernel, a, b; θ = θ,
                                var = var, noise_var = noise_var,
                                jitter = jitter, dims = dims)
        Xdist = broadcast(-, reshape(a, size(a, 1), 1, size(a, 2)),
                             reshape(b, 1, size(b, 1), size(b, 2)))
        K = -0.5 .* (Xdist.^2)
        # K[i, j, k] = -0.5 * (a[i, k] - b[j, k])^2
        # KK[i, j] = var * exp( -0.5 * sum( theta_k (a[i,k]-b[j,k])^2 ) )
        for k in 1:length(θ)
            #ret_k = Dθ_k, n by n
            K_slice = @view K[:, :, k]
            ret_k = Kernel_mat .* K_slice
            push!(ret_set, ret_k)
        end
        return ret_set
    else
        error("failure in evaluate kernel")
    end
end

"""
Evaluate derivative of RBF kernel w.r.t a
"""
function evaluate_derivative_a(kernel::RBFKernelModule, a::Array{T}, b::Array{T};
          θ = kernel.θ, var = kernel.var, noise_var = kernel.noise_var, jitter = 1e-10, dims = 1) where T<:Real
    if size(a, 1) != 1 # for now, only allow vectors
        error("only \"a\" with single row supported")
    end
    if length(θ)!=size(a, 2)
        error("θ and a must have same length")
    end

    k = size(a, 2); m = size(b, 1)
    ret = zeros(k, m)

    sqdistances = zeros(size(a, 1), size(b, 1))

    diff = a .- b
    scale = LinearAlgebra.Diagonal(typeof(θ)<:Array ? θ : [θ])
    dist = diff .* (diff * scale)
    km_no_noise = var[1] * exp.(-0.5 * sum(dist, dims=2))

    ret = - (diff .* km_no_noise) * scale
end

function signal_variance(kernel_module::RBFKernelModule)
    return kernel_module.var
end

function noise_variance(kernel_module::RBFKernelModule)
    return kernel_module.noise_var
end

function inverse_lengthscale(kernel_module::RBFKernelModule)
    return kernel_module.θ
end

# function build_input_dict(t::RBFKernelModule, d)::Dict{a where a<:String, Union{Float64, Array{Float64, 1}, Array{Float64,2}}}
function build_input_dict(t::RBFKernelModule, d)::Dict{String, Any}
    list = ["θ", "var", "noise_var"]
    kernel_dict = Dict()
    for item in list
        if item in keys(d)
            kernel_dict[item] = d[item]
        else
            kernel_dict[item] = t.PARAMETER_DICT[item]
        end
    end
    return kernel_dict
end
