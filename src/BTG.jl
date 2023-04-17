#module BTG

using Pkg
# Pkg.add.(["Tables", "DataFrames", "CSV", "PyPlot", "Plots", "Distributions", "Sobol", "Revise", "IterTools", "FastGaussQuadrature", "StatsFuns"])

using Tables
using DataFrames
using CSV
# using PyPlot
# using Plots
using GaussianProcesses
using Distributions
using Cubature
using TimerOutputs
using Random
using Optim
using Dates

const to = TimerOutput()
# timer locations:
# - bayesian/buffers/caches.jl

include("bayesian/priors/abstract_prior.jl")
include("bayesian/priors/uniform_prior.jl")

include("core/parameters.jl")
include("core/data.jl")
include("transforms/registry.jl")

include("bayesian/buffers/buffers.jl")
include("bayesian/quadrature/quadrature.jl")

include("utils/utils.jl")
include("utils/wrapper.jl")
include("utils/data/get_data.jl")

include("models/model_registry.jl")

include("BTG_simulation/bayesopt/bayesopt.jl")

#end # module
