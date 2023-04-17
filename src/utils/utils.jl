module TypeUtils
    function is_none(x::Any)
        return x == nothing ? true : false
    end
end

include("var_name_to_string.jl")
include("linear_algebra_utils.jl")
include("counter.jl")
include("exp_wrappers.jl")
include("derivative/derivative_checker.jl")
include("data/data.jl")
include("statistics/statistics.jl")
include("statistics/marginal_likelihood.jl")
include("iterator/iterator.jl")
include("iterator/remove_bracket.jl")
