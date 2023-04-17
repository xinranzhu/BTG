include("../../../src/core/data.jl")

function get_random_training_data(;n = 3, d =4, seed = 22)::TrainingData
    rng = MersenneTwister(seed);  # 12345 is the seed
    x = rand(rng, n, d)
    Fx = ones(n, 1)
    y = rand(rng, n)
    return TrainingData(x, Fx, y)
end

"""
Training and testing data generated from noisy sine function
"""
function get_sine_data(; ntrain=101, noise_level=0.1, randseed=1234)
    ntrain = ntrain
    noise_level = noise_level
    rng = MersenneTwister(randseed)
    # sythetic 1d data: y = (sin(x) + 10)^(1/3) + noise
    x = reshape(collect(range(-pi, stop=pi, length=ntrain)), ntrain, 1)
    noise_distribution = Normal(0, sqrt(noise_level))
    noise = rand(rng, noise_distribution, (ntrain, 1))
    sinx_noised = sin.(x) .+ noise
    y = dropdims( sign.(sinx_noised) .* ((abs.(sinx_noised)).^(1/3)) .+ 2, dims=2)
    if minimum(y) < 0
        @warn "Having negative training labels!"
    end
    Fx = ones(ntrain, 1)
    trainingdata0 = TrainingData(x, Fx, y)
    return trainingdata0
end

"""
Generate true label to evaluate prediction
"""
function get_sine_true_label(x0)
    sinx0 = sin.(x0)
    return sign.(sinx0) .* (abs.(sinx0)).^(1/3) .+ 2
end

"""
Generate covariate at new testing point
"""
function get_sine_covariate(x0)
    n_test = size(x0, 1)
    return ones(n_test, 1)
end


function get_int_sine_data(; ntrain=51, noise_level=0.0025, randseed=1234)
    ntrain = ntrain
    noise_level = noise_level
    rng = MersenneTwister(randseed)
    # sythetic 1d data: y = (sin(x) + 10)^(1/3) + noise
    x = reshape(collect(range(-pi, stop=pi, length=ntrain)), ntrain, 1)
    noise_distribution = Normal(0, sqrt(noise_level))
    noise = rand(rng, noise_distribution, (ntrain, 1))
    y = dropdims(round.(sin.(x)) .+ noise .+ 2, dims=2)
    # y = dropdims( sign.(sinx_noised) .* ((abs.(sinx_noised)).^(1/3)) .+ 2, dims=2)
    if minimum(y) < 0
        @warn "Having negative training labels!"
    end
    Fx = ones(ntrain, 1)
    trainingdata0 = TrainingData(x, Fx, y)
    return trainingdata0
end

get_int_sine_true_label(x0) = round.(sin.(x0)) .+ 2

function get_int_sine_covariate(x0)
    n_test = size(x0, 1)
    return ones(n_test, 1)
end



function levy(x)
    w = 1 .+ (x .- 1) ./ 4
    term1 = sinpi(w[1]) ^ 2
    term2 = 0.0
    if length(x)>1
        term2 += sum(z -> (z - 1) ^ 2 * (1 + 10sin(π * z + 1)^2), w[1:end-1])
    end
    term3 = (w[end] - 1) ^ 2 * (1 + sinpi(2w[end]) ^ 2)
    return cbrt((term1 + term2 + term3)-40)+4
end

function get_levy1d_data(; ntrain=51, noise_level=0.0025, randseed=1234)
    rng = MersenneTwister(randseed)
    x = reshape(collect(range(-10, stop=10, length=ntrain)), ntrain, 1)
    noise_distribution = Normal(0, sqrt(noise_level))
    noise = rand(rng, noise_distribution, (ntrain, 1))
    y = dropdims(round.(levy.(x)) .+ noise .+ 2, dims=2)
    Fx = ones(ntrain, 1)
    trainingdata0 = TrainingData(x, Fx, y)
    return trainingdata0
end


#%%
"""
Use f(t) = t^1/3 transformation
"""
function get_nonstationary_data(; ntrain=61, noise_level=0.001, randseed=1234)
    f(x) = -0.5 * (sin(40*(x-0.85)^4)*cos(2.5*( x -0.95))+0.5*(x-0.9) +1) + 0.690
    ntrain = ntrain
    noise_level = noise_level
    rng = MersenneTwister(randseed)
    # sythetic 1d data: y = (sin(x) + 10)^(1/3) + noise
    x = reshape(collect(range(1.0, stop=1.75, length=ntrain)), ntrain, 1)
    noise_distribution = Normal(0, sqrt(noise_level))
    noise = rand(rng, noise_distribution, (ntrain, 1))
    y = dropdims(f.(x), dims=2)
    y = dropdims(cbrt.(y .+ noise),dims=2)
    y = y .+ 1.2
    Fx = ones(ntrain, 1)
    trainingdata0 = TrainingData(x, Fx, y)
    return trainingdata0
end

function get_nonstationary_true_label(x0)
    f(x) = -0.5 * (sin(40*(x-0.85)^4)*cos(2.5*( x -0.95))+0.5*(x-0.9) +1) + 0.690
    y = dropdims(f.(x0), dims=2)
    y = cbrt.(y)
    y = y .+ 1.2
    return y
end

"""
Generate covariate at new testing point
"""
function get_nonstationary_covariate(x0)
    n_test = size(x0, 1)
    return ones(n_test, 1)
end

#%% Gramacy and Lee

function get_gramacy_data(; ntrain=61, noise_level=0.001, randseed=1234)
    f(x) = sin(10*pi*x)./(2*x) + (x-1).^4
    ntrain = ntrain
    noise_level = noise_level
    rng = MersenneTwister(randseed)
    # sythetic 1d data: y = (sin(x) + 10)^(1/3) + noise
    x = reshape(collect(range(0.45, stop=2.3, length=ntrain)), ntrain, 1)
    noise_distribution = Normal(0, sqrt(noise_level))
    noise = rand(rng, noise_distribution, (ntrain, 1))
    y = f.(x)
    if length(size(y))>1
        y = dropdims(y, dims=2)
    end
    y = y .- 0.408 #mean of y in range
    y = cbrt.(y .+ noise)
    if length(size(y))>1
        y = dropdims(y, dims=2)
    end
    y = y .+ 2.8 #make positive
    Fx = ones(ntrain, 1)
    trainingdata0 = TrainingData(x, Fx, y)
    return trainingdata0
end


function get_gramacy_true_label(x0)
    f(x) = sin(10*pi*x)./(2*x) + (x-1).^4
    y = f.(x0)
    if length(size(y))>1
        y = dropdims(y, dims=2)
    end
    y = y .- 0.408 #mean of y in range
    y = cbrt.(y)
    if length(size(y))>1
        y = dropdims(y, dims=2)
    end
    y = y .+ 2.8 #make positive
end

function get_gramacy_covariate(x0)
    n_test = size(x0, 1)
    return ones(n_test, 1)
end
