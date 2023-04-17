struct Id <:AbstractTransform
    PARAMETER_DICT::Dict
    names::Any
    function Id()
        return new(Dict(),[])
    end
end

function evaluate(transform::Id, x::Union{Array{T} where T, G where G<:Real})
    return x
end

function evaluate(t::Id, dict::Dict, y::Real)
    return y
end


function evaluate_inverse(t::Id, dict::Dict, x::Real)
    return x
end


function evaluate_derivative_x(t::Id, dict::Dict, y::Real)
    return 1
end

function evaluate_derivative_hyperparams(t::Id, dict::Dict, y::Real)
    return Dict()
end

function evaluate_derivative_x_hyperparams(t::Id, dict::Dict, y::Real)
    return Dict()
end

