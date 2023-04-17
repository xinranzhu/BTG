using LinearAlgebra

function increment!(b::Buffer, key_dict)
    b[key_dict].times_called = b[key_dict].times_called + 1
end

## Transform cache
transform_cache_timer = TimerOutput()

mutable struct TransformCache <:AbstractCache
    g_of_y::Union{T, Array{T}} where T<:Float64
    function TransformCache(g_of_y)
        return new(g_of_y)
    end
    function TransformCache()
        return new()
    end
end

function construct(
    t::TransformCache,
    transform_dict::Union{Dict, Array{Dict,1}},
    transform::AbstractTransform,
    y::Array{T} where T<:Float64
    )
    g(z) = evaluate(transform, transform_dict, z)
    g_of_y = g.(y)
    return TransformCache(g_of_y)
end

function lookup_or_compute(b::BufferDict, name::String, cache::TransformCache, key_dict::Union{Dict, Array{Dict,1}},
       transform::AbstractTransform, ytrain::Array{T} where T<:Float64; lookup::Bool = true, store = true)::AbstractCache

       if name in keys(b) && key_dict in keys(b[name]) && lookup == true
           @timeit transform_cache_timer "lookup transform cache" begin
               cache_found = b[name][key_dict]
               @assert typeof(cache_found) <: typeof(cache)
           end
           return cache_found
       else #derivatve == true
           @timeit transform_cache_timer "construct transform cache" begin
               if !(name in keys(b)) #b[name] has not been instantiated yet, so the cache certainly hasn't been either
                   b[name] = Buffer()
               end
               #new_cache = construct(cache, key_dict, training_data, kernel_module) #construct cache using key_dict
               new_cache =  construct(cache, key_dict, transform, ytrain) #construct cache using key_dict
               if store == true
                   b[name][key_dict] = new_cache #important to store result
               end
           end
           return new_cache
       end
end

#%% Covariance Matrix cache

covar_cache_timer = TimerOutput()
"""
Implicitly stores kernel matrix formed using kernel_module and training_data.
See RBFKernel for definitions of θ and var.
    - chol, a cholesky decomposition of kernel matrix
"""
mutable struct CovarianceMatrixCache <:AbstractCache
    chol::Cholesky{Float64,Array{Float64, 2}}
    kernel_module::KernelModule
    kernel_derivative_θ::Any #TODO allow for multidimensional θ
    times_called::Int64
    function CovarianceMatrixCache(chol, kernel_module::KernelModule)
        return new(chol, kernel_module, nothing)
    end
    function CovarianceMatrixCache(chol, kernel_module::KernelModule, kernel_derivative_θ)
        return new(chol, kernel_module, kernel_derivative_θ, 1)
    end
    function CovarianceMatrixCache()
        return new()
    end
end

"""
Construct CovarianceMatrixCache using RBFKernel and TrainingData
Note that the ::RBFKernelModule arg is simply a placeholder which directs multiple
dispatch to invoke this method

Args:
    - derivative: bool indicating whether to compute derivative of kernel matrix w.r.t θ
"""
function construct(t::CovarianceMatrixCache, dict::Dict, train::TrainingData,
    ::RBFKernelModule; derivative::Bool = false)::AbstractCache
    #TODO ::Dict{A where A<:String, T} where T<:TNumeric
    @assert "θ" in keys(dict)
    @assert "var" in keys(dict)
    @assert "noise_var" in keys(dict)
    # kernel_module = RBFKernelModule(;θ = dict["θ"], var = dict["var"], noise_var = dict["noise_var"])
    @timeit to "construct CovarianceMatrixCache" begin
    kernel_module = RBFKernelModule(dict)
    #kernel_matrix = evaluate(kernel_module, getposition(train), getposition(train)) #train-train covariance matrix
    kernel_matrix = Symmetric(KernelMatrix(kernel_module, getposition(train), jitter = 1e-10))
    #@show minimum(eigvals(kernel_matrix))
    chol = cholesky(kernel_matrix)
    if derivative == false
        return CovarianceMatrixCache(chol, kernel_module)
    else
        kernel_matrix_derivative_theta  =
        evaluate_derivative_θ(kernel_module, getposition(train), getposition(train),
            jitter = 1e-10)
        return CovarianceMatrixCache(chol, kernel_module, kernel_matrix_derivative_theta)
    end
    end
end

"""
Lookup cache from in BufferDict[name] with keys `key_dict`. If not found, then construct cache and store it in proper place.

Dispatches on cache argument; either looks up cache in BufferDict or computes cache object and stores
it in proper buffer within BufferDict.

N.B. cache is typically an uninstantiated cache -- its type is used to create the instantiated cache. Similarly
kernel_module serves as specification only; the information needed to instantiate it completely is
found in key_dict

Requires:
    - construct function to be defined for buffer_type
"""
function lookup_or_compute(b::BufferDict, name::String, cache::CovarianceMatrixCache, key_dict::Dict{String, T} where T,
       training_data::TrainingData, kernel_module::KernelModule; lookup::Bool = true, derivative::Bool = false, store = true)::AbstractCache
    if lookup == false
        #@warn "Lookup_or_compute is recomputing values each time, because lookup flag is false."
    end
    if name in keys(b) && key_dict in keys(b[name]) && lookup == true && derivative == false
        @timeit covar_cache_timer "lookup covar cache" begin
            increment!(b[name], key_dict)
            cache_found = b[name][key_dict]
            @assert typeof(cache_found) <: typeof(cache)
        end
        return cache_found
    else #derivatve == true
        @timeit covar_cache_timer "construct covar cache" begin
            if !(name in keys(b)) #b[name] has not been instantiated yet, so the cache certainly hasn't been either
                new_buffer = Buffer() #empty cache buffer
                b[name] = new_buffer
            end
            #new_cache = construct(cache, key_dict, training_data, kernel_module) #construct cache using key_dict
            new_cache =  construct(cache, key_dict, training_data, kernel_module, derivative = derivative) #construct cache using key_dict
            if derivative == false && store == true
                b[name][key_dict] = new_cache #important to store result
            end
        end
        return new_cache
    end
end

#%% Training Cache

train_cache_timer = TimerOutput()

mutable struct TrainingCache <:AbstractCache
    Σθ_inv_y::Array{T} where T<:Float64
    times_called::Int64
    function TrainingCache(Σθ_inv_y)
        return new(Σθ_inv_y, 1)
    end
    function TrainingCache()
        return new()
    end
end

"""
    Has dependency on CovarianceMatrixCache. It is not always the case that a construct
    function must take the global BufferDict as an arg, but in this case we need cholΣθ,
    in case it has already been computed
    Args:
        - g_of_y: nonlinear transformed y
"""
function construct(t::TrainingCache, dict::Dict, training_data::TrainingData, ::RBFKernelModule,
    covariance_buffer_name::String, g_of_y, buffer::BufferDict)::AbstractCache
    @assert "θ" in keys(dict)
    @assert "var" in keys(dict)
    @assert "noise_var" in keys(dict)
    @timeit to "construct TrainingCache" begin
        covariance_key_dict = build_input_dict(RBFKernelModule(), dict)
        kernel_module = RBFKernelModule(covariance_key_dict)
        covariance_matrix_cache = lookup_or_compute(buffer, covariance_buffer_name,
            CovarianceMatrixCache(), covariance_key_dict, training_data, kernel_module)
        chol = covariance_matrix_cache.chol
        Σθ_inv_y = chol\g_of_y
        return TrainingCache(Σθ_inv_y)
    end
end


function lookup_or_compute(buffer::BufferDict, name::String, cache::TrainingCache,
    key_dict::Dict, training_data::TrainingData,
    kernel_module::KernelModule, g_of_y, covariance_buffer_name::String; lookup = true)::AbstractCache
    if lookup == false
        #@warn "Lookup_or_compute is recomputing values each time, because lookup flag is false."
    end
    if name in keys(buffer) && key_dict in keys(buffer[name]) && lookup == true
        @timeit covar_cache_timer "lookup TrainingCache" begin
            increment!(buffer[name], key_dict)
            cache_found = buffer[name][key_dict]
            @assert typeof(cache_found) <: typeof(cache)
            return cache_found
        end
    else
        @timeit covar_cache_timer "construct TrainingCache" begin
            if !(name in keys(buffer)) #b[name] has not been instantiated yet, so the cache certainly hasn't been either
                new_buffer = Buffer() #empty cache buffer
                buffer[name] = new_buffer
            end
            # @warn "Constructing new training cache instead of using lookup..."
            new_cache = construct(TrainingCache(), key_dict, training_data, kernel_module,
                covariance_buffer_name, g_of_y, buffer)
            buffer[name][key_dict] = new_cache #important to store result
        end
        return new_cache
    end
end

#%% Testing Cache
 testing_cache_timer = TimerOutput()

mutable struct TestingCache <:AbstractCache
    Bθ::Array{T, 2} where T<:Float64
    Eθ::Array{T, 2} where T<:Float64 #test-test kernel matrix
    Dθ::Array{T, 2} where T<:Float64
    Bθ_prime_x
    Eθ_prime_x
    Dθ_prime_x
    times_called::Int64
    function TestingCache(Bθ, Eθ, Dθ, Bθ_prime_x, Eθ_prime_x, Dθ_prime_x)
        return new(Bθ, Eθ, Dθ, Bθ_prime_x, Eθ_prime_x, Dθ_prime_x, 1)
    end
    function TestingCache(Bθ, Eθ, Dθ)
        return new(Bθ, Eθ, Dθ, 1)
    end
    function TestingCache()
        return new()
    end
end

"""
Construct TestingCache
Args:
    dict: contains keys "θ", "var", "x0", where the first two are kernel matrix
          hyperparameters and x0 is the location of the test point
    covariance_buffer_name: name of covariance_buffer to look up in buffer,
          if Σθ is needed
"""
function construct(t::TestingCache, dict::Dict, training_data::TrainingData,
    ::RBFKernelModule, covariance_buffer_name::String, buffer_dict::BufferDict;
    derivatives = false)::AbstractCache
    @timeit testing_cache_timer "assertions" begin
        @assert "θ" in keys(dict)
        @assert "var" in keys(dict)
        @assert "noise_var" in keys(dict)
        @assert "x0" in keys(dict)
    end
    #@timeit to "construct TestingCache" begin
    x0 = dict["x0"]
    if typeof(x0) == Float64
        x0 = reshape([dict["x0"]], 1, getdimension(training_data))
    end
    x = getposition(training_data)
    @timeit testing_cache_timer "cov input dict" begin
        covariance_dict = build_input_dict(RBFKernelModule(), dict)
    end
    @timeit testing_cache_timer "define kernel_module" begin
        kernel_module = RBFKernelModule(covariance_dict)
    end
    @timeit testing_cache_timer "lookup cholKθ" begin
        covariance_matrix_cache = lookup_or_compute(buffer_dict, covariance_buffer_name,
            CovarianceMatrixCache(), covariance_dict, training_data, kernel_module)
    end
    @timeit testing_cache_timer "access cholKθ" begin
        chol = covariance_matrix_cache.chol
    end
    #@info x0
    #@info x
    @timeit testing_cache_timer "compute Bθ" begin
        Bθ = KernelMatrix(kernel_module, x0, x, noise_var = 0.0) #turn off observation noise
    end
    @timeit testing_cache_timer "compute Eθ" begin
        Eθ = KernelMatrix(kernel_module, x0; noise_var = 0.0, jitter = 1e-10) #no noise here!
    end
    @timeit testing_cache_timer "compute Dθ" begin
        Dθ = Eθ - Bθ*(chol\Bθ') .+ covariance_dict["noise_var"][1] .+  1e-10 # add noise and jitter
    end
    if derivatives == false
        return TestingCache(Bθ, Eθ, Dθ)
    else
        #column vector
        @timeit testing_cache_timer "derivatives" begin
            Bθ_prime_x = reshape(evaluate_derivative_a(kernel_module, x0, x), getnumpoints(training_data), length(x0)) #derivative w.r.t location x0
            Eθ_prime_x = reshape([0], 1, 1) #constant independent of position
            Dθ_prime_x = -  2*Bθ_prime_x'*(chol\Bθ') #- Bθ*(chol\Bθ_prime_x)
        end
        return TestingCache(Bθ, Eθ, Dθ, Bθ_prime_x, Eθ_prime_x, Dθ_prime_x)
    end
    #end
end

"""
Lookup or compute TestingCache. If testing_cache corresponding to key_dict
is not found in buffer (a dictionary), then constructs cache and places it in
buffer.
    Args:
        - buffer: dictionary of TestingCaches
"""
function lookup_or_compute(buffer::Buffer, buffer_dict::BufferDict, cache::TestingCache,
    key_dict::Dict where T, training_data::TrainingData,
    kernel_module::KernelModule, covariance_buffer_name::String; lookup = true, derivatives = false)::AbstractCache
    if lookup == false
        #@warn "Lookup_or_compute is recomputing values each time, because lookup flag is false."
    end
    if key_dict in keys(buffer) && lookup == true
        @timeit testing_cache_timer "lookup TestingCache" begin
            increment!(buffer, key_dict)
            cache_found = buffer[key_dict]
            @assert typeof(cache_found) <: typeof(cache)
            return cache_found
        end
    else
        #@info "calculating in lookup_or_compute for test_buffer"
        @timeit testing_cache_timer "construct test cache" begin
            new_cache = construct(TestingCache(), key_dict, training_data,
                    kernel_module, covariance_buffer_name, buffer_dict; derivatives = derivatives)

            buffer[key_dict] = new_cache #important to store result
        end
        return new_cache
    end
end
