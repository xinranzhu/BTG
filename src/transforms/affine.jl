"""
Affine transformation from R^1 to R^1
g(y) = a + by
"""

mutable struct Affine{T} <:AbstractTransform
    a::Union{T, Nothing}
    b::Union{T, Nothing}
    PARAMETER_DICT::Dict{String, Union{T, Nothing}} #dictionary representation of parameters
    names::Array{W} where W<:String
    Affine(a, b; my_type=Real) = new{my_type}(a,b, Dict("a"=>a, "b"=>b), ["a", "b"])
    function Affine(dict; my_type=Real)
        names = ["a", "b"]
        if !issubset(Set(keys(dict)), Set(names))
            error("initializer dict d must have keys in {\"a\", \"b\"}")
        end
        a = get(dict, "a", nothing)
        b = get(dict, "b", nothing)
        dict_new = Dict("a"=>a, "b"=>b)
        new{my_type}( a,  b, dict_new, names)
    end
    Affine() = new{Float64}(nothing, nothing, Dict("a"=>nothing, "b"=>nothing), ["a", "b"])
end

#%% Helpers

# getParameterDict(t::Affine)::Dict = t.PARAMETER_DICT #TODO type check Dict

# function getArgs(t::Affine, dict::Dict) #TODO what's the return type here?
#     if isdefined(t, 2) #check PARAMETER_DICTIONARY is defined
#         a = "a" in keys(dict) ? dict["a"] : ("a" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["a"] : error("a not supplied or defined"))
#         b = "b" in keys(dict) ? dict["b"] : ("b" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["b"] : error("b not supplied or defined"))
#     else
#         @assert Set(keys(dict)) == Set(["a", "b"])
#     end
#     a = length(a) == 1 ? a[1] : a
#     b = length(b) == 1 ? b[1] : b
#     return (a, b)
# end

#%% Evaluation

evaluate(t::Affine, y::Real) = t.a + t.b * y

function evaluate(t::Affine, a, b, y::Real)
    #TODO type check a, b
    return a + b * y
end

"""
Evaluate t with dictionary of inputs -- possibly with partial values.
Order of precedence in searching for "a" is d, then t.PARAMETER_DICT, then t.a
"""
function evaluate(t::Affine, dict::Dict, y::Real)
    (a, b) = getArgs(t, dict)
    return a + b * y
end

#%% Derivatives

evaluate_derivative_x(t::Affine, y::Real) = t.b

function evaluate_derivative_x(t::Affine, dict::Dict, y::Real)
    (a, b) = getArgs(t, dict)
    return b
end

function evaluate_derivative_x2(t::Affine, dict::Dict, y::Real)
    return 0.
end

#%% Inverse
evaluate_inverse(t::Affine, x::Real) = (x - t.a)/t.b

function evaluate_inverse(t::Affine, d::Dict, x::Real)
    (a, b) = getArgs(t, d)
    return (x - a)/b
end


function evaluate_derivative_hyperparams(t::Affine, dict::Dict, y::Real)
    derivative_dict = Dict()
    (a, b) = getArgs(t, dict)
    for key in keys(dict)
        if key == "a"
            derivative_dict["a"] = 1
        elseif key == "b"
            derivative_dict["b"] = y
        end
    end
    return derivative_dict
end



function evaluate_derivative_x_hyperparams(t::Affine, dict::Dict, y::Real)
    derivative_dict = Dict()
    (a, b) = getArgs(t, dict)
    for key in keys(dict)
        if key == "a"
            derivative_dict["a"] = 0.
        elseif key == "b"
            derivative_dict["b"] = 1.
        end
    end
    return derivative_dict
end
