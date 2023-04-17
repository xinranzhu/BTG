"""
Monotonic warping function From R^1 to R^1
Reference:
https://www.jstor.org/stable/pdf/27798865.pdf
Math:
    g(y) = sinh(b* SinhArcSinh(y) - a)
Parameters:
    a, b: real number
"""
mutable struct SinhArcSinh{T} <:AbstractTransform
    a::Union{T, Nothing}
    b::Union{T, Nothing}
    PARAMETER_DICT::Dict{String, Union{T, Nothing}} #dictionary representation of parameters
    names::Array{W} where W<:String
    SinhArcSinh(a, b; my_type=Real) = new{my_type}(a, b, Dict("a"=>a, "b"=>b), ["a", "b"])
    """
    We can initialize an SinhArcSinh object with a complete or incomplete dictionary
    """
    function SinhArcSinh(dict; my_type=Real)
        names = ["a", "b"]
        if !issubset(Set(keys(dict)), Set(names))
            error("initializer dict must have keys in {\"a\", \"b\"}")
        end
        a = get(dict, "a", nothing)
        b = get(dict, "b", nothing)
        dict_new = Dict("a"=>a, "b"=>b)
        new{my_type}( a,  b, dict_new, names)
    end
    SinhArcSinh() = new{Float64}(nothing, nothing, Dict("a"=>nothing, "b"=>nothing), ["a", "b"])
end

#%% Helpers
# getParameterDict(t::SinhArcSinh)::Dict = t.PARAMETER_DICT #TODO type check Dict

# function getArgs(t::SinhArcSinh, dict::Dict) #TODO what's the return type here?
#     if isdefined(t, 3) #check PARAMETER_DICTIONARY is defined
#         a = "a" in keys(dict) ? dict["a"] : ("a" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["a"] : error("a not supplied or defined"))
#         b = "b" in keys(dict) ? dict["b"] : ("b" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["b"] : error("b not supplied or defined"))
#     else #case of uninitialized transform
#         @assert Set(keys(dict)) == Set(["a", "b"])
#         a = dict["a"]; b = dict["b"]
#     end
#     a = length(a) == 1 ? a[1] : a
#     b = length(b) == 1 ? b[1] : b
#     return (a, b)
# end

#%% evaluate

evaluate(t::SinhArcSinh, y::Real) = sinh(t.b * asinh(y) - t.a)

function evaluate(t::SinhArcSinh, d::Dict, y::Real)
    #TODO type check a, b
    (a, b) = getArgs(t, d)
    return sinh(b * asinh(y) - a)
end

function evaluate(t::SinhArcSinh, a, b, y::Real)
    #TODO type check a, b
    if typeof(a) == Array{Float64,1}
        a = a[1]
    end
    if typeof(b) == Array{Float64,1}
        b = b[1]
    end
    return sinh(b * asinh(y) - a)
end



#%% Derivatives

evaluate_derivative_x(t::SinhArcSinh, y::Real) = (t.b * cosh(t.b*asinh(y)-t.a)) / sqrt(1+y^2)

function evaluate_derivative_x(t::SinhArcSinh, d::Dict, y::Real) #TODO parametrize y
    (a, b) = getArgs(t, d)
    return (b * cosh(b*asinh(y)-a)) / sqrt(1+y^2)
end

function evaluate_derivative_x2(t::SinhArcSinh, d::Dict, y::Real) #TODO parametrize y
    (a, b) = getArgs(t, d)
    constant = b/(1+y^2)
    tmp = b * asinh(y) - a
    factor1 = sinh(tmp) * b
    factor2 = cosh(tmp) * y / sqrt(1+y^2) / (1+y^2)
    return constant * (factor1 + factor2)
end

#%% Inverse
evaluate_inverse(t::SinhArcSinh, d::Dict, x::Real) = ((a, b) = getArgs(t, d); sinh( (asinh(x) + a) / b))

function evaluate_inverse(t::SinhArcSinh, d::Dict, x::Real)
    (a, b) = getArgs(t, d)
    return sinh((asinh(x)+a) / b)
end

function evaluate_derivative_hyperparams(t::SinhArcSinh, dict::Dict, y::Real)
    derivative_dict = Dict()
    (a, b) = getArgs(t, dict)
    tmp = cosh( b*asinh(y) - a)
    for key in keys(dict)
        if key == "a"
            derivative_dict["a"] = - tmp
        elseif key == "b"
            derivative_dict["b"] = tmp * asinh(y)
        end
    end
    return derivative_dict
end


function evaluate_derivative_x_hyperparams(t::SinhArcSinh, dict::Dict, y::Real)
    derivative_dict = Dict()
    (a, b) = getArgs(t, dict)
    tmp = sqrt(1+y^2)
    tmp2 = b*asinh(y) - a
    for key in keys(dict)
        if key == "a"
            derivative_dict["a"] = (- b * sinh(tmp2))/ tmp
        elseif key == "b"
            derivative_dict["b"] = (cosh(tmp2) + asinh(y) * b * sinh(tmp2)) / tmp
        end
    end
    return derivative_dict
end
