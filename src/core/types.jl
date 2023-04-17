#%% Define commonly used types

"""
Array of Float64 or Float64
"""
const TNumeric = Union{Array{T}, T} where T<: Float64
const TArray = Array{T} where T<:Float64
