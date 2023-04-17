abstract type AbstractData end
abstract type AbstractTestingData <: AbstractData end
abstract type AbstractTrainingData <:AbstractData end
include("testing_data.jl")
include("training_data.jl")
