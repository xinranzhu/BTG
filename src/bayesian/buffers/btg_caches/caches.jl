using LinearAlgebra

#%% BTG Training Cache
mutable struct BTGCovarianceCache<:AbstractCache
    kernel_module::KernelModule
    Σθ_inv_X::Array{Float64, 2}
    choleskyΣθ::Cholesky
    choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}
    logdetΣθ::Float64
    logdetXΣX::Float64
    dΣθ::Any
    n::Int64 #number data points incorporate
    function BTGCovarianceCache(kernel_module, Σθ_inv_X, choleskyΣθ, choleskyXΣX, dΣθ)
        logdetΣθ = logdet(choleskyΣθ)
        logdetXΣX = logdet(choleskyXΣX)
        n = size(choleskyΣθ, 1)
        return new(kernel_module, Σθ_inv_X, choleskyΣθ, choleskyXΣX, logdetΣθ, logdetXΣX, dΣθ, n)
    end
    function BTGCovarianceCache()
        return new()
    end
end

function  construct(t::BTGCovarianceCache, dict::Dict, train::TrainingData,
    ::RBFKernelModule; derivative::Bool = false)::AbstractCache
    @assert "θ" in keys(dict)
    @assert "var" in keys(dict)
    @assert "noise_var" in keys(dict)

    kernel_module = RBFKernelModule(dict)
    kernel_matrix = Symmetric(KernelMatrix(kernel_module, getposition(train), jitter = 1e-10))
    choleskyΣθ = cholesky(kernel_matrix)
    dΣθ = nothing
    if derivative == true
        dΣθ  = evaluate_derivative_θ(kernel_module, getposition(train), getposition(train), jitter = 1e-10)
    end
    Fx = getcovariate(train)
    Σθ_inv_X = choleskyΣθ\Fx
    XΣX = Fx'*Σθ_inv_X
    choleskyXΣX = cholesky(Symmetric(XΣX))
    return BTGCovarianceCache(kernel_module, Σθ_inv_X, choleskyΣθ, choleskyXΣX, dΣθ)
end

function lookup_or_compute(b::BufferDict, name::String, cache::BTGCovarianceCache, key_dict::Dict{String, T} where T,
       training_data::TrainingData, kernel_module::KernelModule; lookup::Bool = true, derivative::Bool = false, store = true)::AbstractCache
    if lookup == false
        #@warn "Lookup_or_compute is recomputing values each time, because lookup flag is false."
    end
    if name in keys(b) && key_dict in keys(b[name]) && lookup == true && derivative == false
        cache_found = b[name][key_dict]
        @assert typeof(cache_found) <: typeof(cache)
        return cache_found
    else #derivatve == true
        if !(name in keys(b)) #b[name] has not been instantiated yet, so the cache certainly hasn't been either
            new_buffer = Buffer() #empty cache buffer
            b[name] = new_buffer
        end
        #new_cache = construct(cache, key_dict, training_data, kernel_module) #construct cache using key_dict
        new_cache =  construct(cache, key_dict, training_data, kernel_module, derivative = derivative) #construct cache using key_dict
        if derivative == false && store == true
            b[name][key_dict] = new_cache #important to store result
        end
        return new_cache
    end
end

#%% BTGTrainingCache
"""
Computations with transformed training observations, depends on transorm parameters,
May call lookup_or_compute on BTGCovarianceCache to obtain cholΣθ
"""
mutable struct BTGTrainingCache<:AbstractCache
    βhat::Array{T} where T<:Real
    qtilde::Real
    Σθ_inv_y::Array{T} where T<:Real #inv g of y
    function BTGTrainingCache(βhat, qtilde, Σθ_inv_y)
        return new(βhat, qtilde, Σθ_inv_y)
    end
    function BTGTrainingCache()
        return new()
    end
end

function construct(t::BTGTrainingCache, dict::Dict, training_data::TrainingData, ::RBFKernelModule,
    covariance_buffer_name::String, g_of_y, buffer::BufferDict)::BTGTrainingCache
    @assert "θ" in keys(dict)
    @assert "var" in keys(dict)
    @assert "noise_var" in keys(dict)
    Fx = getcovariate(training_data)
    covariance_key_dict = build_input_dict(RBFKernelModule(), dict)
    kernel_module = RBFKernelModule(covariance_key_dict)
    covariance_matrix_cache = lookup_or_compute(buffer, covariance_buffer_name,
        BTGCovarianceCache(), covariance_key_dict, training_data, kernel_module)
    choleskyΣθ = covariance_matrix_cache.choleskyΣθ
    choleskyXΣX = covariance_matrix_cache.choleskyXΣX
    Σθ_inv_X = covariance_matrix_cache.Σθ_inv_X
    #computations
    Σθ_inv_y = choleskyΣθ\g_of_y
    βhat = choleskyXΣX\(Fx'*Σθ_inv_y)
    qtilde =  g_of_y'*Σθ_inv_y  - 2*g_of_y'*Σθ_inv_X*βhat + βhat'*Fx'*Σθ_inv_X*βhat
    return BTGTrainingCache(βhat, qtilde[1], Σθ_inv_y)
end


function lookup_or_compute(buffer::BufferDict, name::String, cache::BTGTrainingCache,
    key_dict::Dict, training_data::TrainingData,
    kernel_module::KernelModule, g_of_y, covariance_buffer_name::String; lookup = true)::BTGTrainingCache
    if lookup == false
        #@warn "Lookup_or_compute is recomputing values each time, because lookup flag is false."
    end
    if name in keys(buffer) && key_dict in keys(buffer[name]) && lookup == true
        @timeit to "lookup TrainingCache" begin
            cache_found = buffer[name][key_dict]
            @assert typeof(cache_found) <: typeof(cache)
            return cache_found
        end
    else
        if !(name in keys(buffer)) #b[name] has not been instantiated yet, so the cache certainly hasn't been either
            new_buffer = Buffer() #empty cache buffer
            buffer[name] = new_buffer
        end
        new_cache = construct(BTGTrainingCache(), key_dict, training_data, kernel_module,
            covariance_buffer_name, g_of_y, buffer)
        buffer[name][key_dict] = new_cache #important to store result
        return new_cache
    end
end


#%% BTG Testing Cache

mutable struct BTGTestingCache<:AbstractCache
    Eθ::Union{Array{Float64, 2}, Nothing}
    Bθ::Union{Array{Float64, 2}, Nothing}
    ΣθinvBθ::Union{Array{Float64, 2}, Nothing}
    Dθ::Union{Array{Float64, 2}, Nothing}
    Hθ::Union{Array{Float64, 2}, Nothing}
    Cθ::Union{Array{Float64, 2}, Nothing}
    function BTGTestingCache()
        return new()
    end
    function BTGTestingCache(Eθ, Bθ, ΣθinvBθ, Dθ, Hθ, Cθ)
        return new(Eθ, Bθ, ΣθinvBθ, Dθ, Hθ, Cθ)
    end
end

function construct(t::BTGTestingCache, dict::Dict, training_data::TrainingData,
    ::RBFKernelModule, covariance_buffer_name::String, buffer_dict::BufferDict)::BTGTestingCache
    @assert "θ" in keys(dict)
    @assert "var" in keys(dict)
    @assert "noise_var" in keys(dict)
    @assert "x0" in keys(dict)
    @assert "Fx0" in keys(dict)
    x0 = dict["x0"]
    Fx0 = dict["Fx0"]
    covariance_dict = build_input_dict(RBFKernelModule(), dict)
    kernel_module = RBFKernelModule(covariance_dict)
    covariance_matrix_cache = lookup_or_compute(buffer_dict, covariance_buffer_name,
        BTGCovarianceCache(), covariance_dict, training_data, kernel_module)
    choleskyΣθ = covariance_matrix_cache.choleskyΣθ
    Σθ_inv_X = covariance_matrix_cache.Σθ_inv_X
    choleskyXΣX = covariance_matrix_cache.choleskyXΣX
    Bθ = KernelMatrix(kernel_module, x0, getposition(training_data), noise_var = 0.0) #turn off observation noise
    Eθ = KernelMatrix(kernel_module, x0; noise_var = 0.0, jitter = 1e-10) #no noise here!
    Dθ = Eθ - Bθ*(choleskyΣθ\Bθ') .+ covariance_dict["noise_var"][1] .+  1e-10 # add noise and jitter
    Hθ = Fx0 - Bθ*Σθ_inv_X
    Cθ = Dθ + Hθ*(choleskyXΣX\Hθ')
    ΣθinvBθ = choleskyΣθ\Bθ'
    return BTGTestingCache(Eθ, Bθ, ΣθinvBθ, Dθ, Hθ, Cθ)
end


function lookup_or_compute(buffer::Buffer, buffer_dict::BufferDict, cache::BTGTestingCache,
    key_dict::Dict where T, training_data::TrainingData,
    kernel_module::KernelModule, covariance_buffer_name::String; lookup = true)::BTGTestingCache
    if lookup == false
        #@warn "Lookup_or_compute is recomputing values each time, because lookup flag is false."
    end
    if key_dict in keys(buffer) && lookup == true
        cache_found = buffer[key_dict]
        @assert typeof(cache_found) <: typeof(cache)
        return cache_found
    else
        #@info "calculating in lookup_or_compute for test_buffer"
        new_cache = construct(BTGTestingCache(), key_dict, training_data,
                kernel_module, covariance_buffer_name, buffer_dict)
        buffer[key_dict] = new_cache #important to store result
        return new_cache
    end
end
