import Base: merge
abstract type ParameterType end
struct Parameter <: ParameterType
    name::String
    dimension::Int64
    range::Array{Float64}
    prior::Union{AbstractPrior, Nothing}
    log_scale::Bool
    function Parameter(name::String, dimension::Int, range::Array{T} where T<:Real; prior=nothing, log_scale = false)::Parameter
        @assert size(range)[2] == 2 "Only range with two ends is accepted"
        @assert dimension == size(range)[1] "Dimension of parameter and range should match"
        @assert minimum(range[:, 2] .> range[:, 1]) "Range should start with minimum and end with maximum in each dimension"
        return new(name, dimension, range, prior, log_scale)
    end
    function Parameter(name::String)::Parameter
        return new(name, 1, reshape([0, 1], 1, 2), nothing, false)
    end
end

getname(p::Parameter)::String = p.name
getdimension(p::Parameter)::Int64 = p.dimension
getrange(p::Parameter)::Array{Float64} = p.range
getprior(p::Parameter)::Union{AbstractPrior, Nothing} = p.prior

function merge(parameters::Array{Parameter, 1})::Parameter
    name = "MERGED_PARAMETER"
    range = reduce( (x, y) -> cat(x, y, dims = 1), [getrange(x) for x in parameters])
    dimension = reduce(+, [getdimension(x) for x in parameters])
    return Parameter(name, dimension, range)
end
