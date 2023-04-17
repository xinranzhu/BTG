include("kernel_module.jl")

"""
Forms explicit kernel matrix using specs drawn out in KernelModule
"""
function KernelMatrix(km::KernelModule, x::Array{Float64, 2}, y::Array{Float64, 2};
    θ = km.θ, var = km.var, noise_var = km.noise_var)
    return evaluate(km, x, y; θ = θ, var = var, noise_var = noise_var)
end

function KernelMatrix(km::KernelModule, x::Array{Float64, 2}; θ = km.θ, var = km.var,
    noise_var = km.noise_var, jitter = 1e-10)
    return evaluate(km, x, x; θ = θ, var = var, noise_var = noise_var, jitter = jitter)
end
