#%%
"""
Monotonic warping function From R^1 to R^1
Resulting from passing a N(0,1) random variable through the inverse hyperbolic sine transformation, known as the Johnson's SU-distribution
Reference:
https://www.jstor.org/stable/pdf/2332539.pdf
https://reader.elsevier.com/reader/sd/pii/S0893608019301856?token=CE1CC494E1E6D2DA4778B71951583295DCF9ACD6E48BFB2532BBBD852BA79F46F6D53E2FE4EA6FB790AAB7C09E51492D
Math:
    g(y) = a + b arcsinh((y-c)/d)
Parameters:
    a, b: real number
    c, d: non-negative real number
Note: Set type T to Float64 is optimal for computational efficiency.
"""
mutable struct ArcSinh{T} <:AbstractTransform
    a::Union{T, Nothing}
    b::Union{T, Nothing}
    c::Union{T, Nothing}
    d::Union{T, Nothing}
    PARAMETER_DICT::Dict{String, Union{T, Nothing}} #dictionary representation of parameters
    names::Array{W} where W<:String
    ArcSinh(a, b, c, d; my_type=Real) = (@assert c>=0; @assert d>=0;
                    new{my_type}(a,b,c,d, Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d), ["a", "b", "c", "d"]))
    """
    We can initialize an ArcSinh object with a complete or incomplete dictionary
    """
    function ArcSinh(dict; my_type=Real)
        names = ["a", "b", "c", "d"]
        if !issubset(Set(keys(dict)), Set(names))
            error("initializer dict d must have keys in {\"a\", \"b\", \"c\", \"d\"}")
        end
        a = get(dict, "a", nothing)
        b = get(dict, "b", nothing)
        c = get(dict, "c", nothing)
        d = get(dict, "d", nothing)
        dict_new = Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d)
        new{my_type}(a, b, c, d, dict_new, names)
    end
    ArcSinh() = new{Float64}(nothing, nothing, nothing, nothing, Dict("a"=>nothing, "b"=>nothing, "c"=>nothing, "d"=>nothing),
    ["a", "b", "c", "d"])
end

# #%% Helper Functions
# getParameterDict(t::ArcSinh)::Dict = t.PARAMETER_DICT #TODO type check Dict

# """
# Helper function for evaluate(Arcsinh, d) where d is incomplete dictionary input.
# Values in d are given precedence.
# GetArgs collects variables to be used in transformation, first by looking
# inside d for variable values and then the fixed parameters in t. This function
# is necessary because d may only contain partial information about parameter values.
# """
# function getArgs(t::ArcSinh, dict::Dict) #TODO what's the return type here?

#     if isdefined(t, 5) #check PARAMETER_DICTIONARY is defined
#         a = "a" in keys(dict) ? dict["a"] : ("a" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["a"] : error("a not supplied or defined"))
#         b = "b" in keys(dict) ? dict["b"] : ("b" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["b"] : error("b not supplied or defined"))
#         c = "c" in keys(dict) ? dict["c"] : ("c" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["c"] : error("c not supplied or defined"))
#         d = "d" in keys(dict) ? dict["d"] : ("d" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["d"] : error("d not supplied or defined"))
#     else
#         @assert Set(keys(dict)) == Set(["a", "b", "c", "d"])
#         a = dict["a"]; b = dict["b"]; c = dict["c"]; d = dict["d"]
#     end
#     a = length(a) == 1 ? a[1] : a
#     b = length(b) == 1 ? b[1] : b
#     c = length(a) == 1 ? c[1] : c
#     d = length(b) == 1 ? d[1] : d
#     return (a, b, c, d)
# end

#%% Evaluate
evaluate(t::ArcSinh{T} where T, y) = (t.a + t.b * asinh((y-t.c)/t.d))

function evaluate(t::ArcSinh, a, b, c, d, y::Real)
    #TODO type check a, b, c, d
    return a + b * asinh((y-c)/d)
end

"""
Evaluate t with dictionary of inputs -- possibly with partial values.
Order of precedence in searching for "a" is d, then t.PARAMETER_DICT, then t.a
"""
function evaluate(t::ArcSinh, dict::Dict, y::Real)
    (a, b, c, d) = getArgs(t, dict)
    return a + b * asinh((y-c)/d)
end

#%% Evaluate Derivatives
evaluate_derivative_x(t::ArcSinh, y::Real) = t.b / sqrt(t.d^2 + (y-t.c)^2)

function evaluate_derivative_x(t::ArcSinh, dict::Dict, y::Real)
    (a, b, c, d) = getArgs(t, dict)
    return b / sqrt(d^2 + (y-c)^2)
end

function evaluate_derivative_x2(t::ArcSinh, dict::Dict, y::Real)
    (a, b, c, d) = getArgs(t, dict)
    tmp = sqrt(d^2 + (y-c)^2)
    return -b * (y-c) / tmp^3
end

#%% Evaluate Inverse
evaluate_inverse(t::ArcSinh, dict::Dict, x::Real) = ( (a,b,c,d)=getArgs(t, dict); c + d * sinh((x - a) / b) )
function evaluate_inverse(t::ArcSinh, dict::Dict, x::Real)
    (a, b, c, d) = getArgs(t, dict)
    return c + d * sinh((x - a) / b)
end


function evaluate_derivative_hyperparams(t::ArcSinh, dict::Dict, y::Real)
    derivative_dict = Dict()
    (a, b, c, d) = getArgs(t, dict)
    tmp = (y-c)/d
    for key in keys(dict)
        if key == "a"
            derivative_dict["a"] = 1.0
        elseif key == "b"
            derivative_dict["b"] = asinh(tmp)
        elseif key == "c"
            derivative_dict["c"] = - b / sqrt(tmp^2+1) / d
        elseif key == "d"
            derivative_dict["d"] =  - b * tmp / sqrt(tmp^2+1) / d
        end
    end
    return derivative_dict
end



function evaluate_derivative_x_hyperparams(t::ArcSinh, dict::Dict, y::Real)
    derivative_dict = Dict()
    (a, b, c, d) = getArgs(t, dict)
    tmp = d^2 + (y-c)^2
    # tmp2 = b*asinh(y) - a
    for key in keys(dict)
        if key == "a"
            derivative_dict["a"] = 0
        elseif key == "b"
            derivative_dict["b"] = 1/sqrt(tmp)
        elseif key == "c"
            derivative_dict["c"] = b*(y-c)/tmp/sqrt(tmp)
        elseif key == "d"
            derivative_dict["d"] = - b*d/tmp/sqrt(tmp)
        end
    end
    return derivative_dict
end
