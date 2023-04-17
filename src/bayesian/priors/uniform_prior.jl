
mutable struct UniformPrior
    range::Array{T} where T<:Real
end

function evaluate(prior::UniformPrior, x::Union{Array{T}, T} where T<:Real; log_scale=false)
    @assert prod(range[:, 1] .<= x && x .<= range[:, 2]) == 1
    difference = range[:, 2] .- range[:, 1]
    return log_scale ? -sum(log.(difference)) : 1/prod(difference)
end


