"""
Monotonic warping function From R^1 to R^1
Math:
    g(y) = (y^λ - 1)/λ, where λ>0
    g(y) = log(y), where λ=0
Parameters:
    λ: nonnegative real number
"""
struct BoxCox <:AbstractTransform
    λ::Union{T, Nothing} where T<:Real
    PARAMETER_DICT::Dict{String, Any}
    names::Array{W} where W<:String
    BoxCox(λ) = new(λ, Dict("λ"=>λ), ["λ"])
    function BoxCox(dict)
        names =  ["λ"]
        if !issubset(Set(keys(dict)), Set(names))
            error("initializer dict must have keys in {\"λ\"}")
        end
        λ = get(dict, "λ", nothing)
        dict_new = Dict("λ"=>λ)
        new(λ, dict_new, names)
    end
    BoxCox() = new(nothing, Dict("λ"=>nothing), ["λ"])
end

#%% Helpers
# getParameterDict(t::BoxCox)::Dict = t.PARAMETER_DICT #TODO type check Dict


# function getArgs(t::BoxCox, dict::Dict) #TODO what's the return type here?
#     if isdefined(t, 2) #check PARAMETER_DICTIONARY is defined
#         λ = "λ" in keys(dict) ? dict["λ"] : ("λ" in keys(t.PARAMETER_DICT) ? t.PARAMETER_DICT["λ"] : error("λ not supplied or defined"))
#     else #case of uninitialized transform
#         @assert Set(keys(dict)) == Set(["λ"])
#         λ = dict["λ"]
#     end
#     λ = length(λ) == 1 ? λ[1] : λ
#     return λ
# end



function evaluate(t::BoxCox, y::Real)
    λ = t.λ
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1] == 0 ? log.(y) : expm1.(log.(y) .* λ[1]) ./ λ[1]
    return out
end
function evaluate(t::BoxCox, d::Dict, y::Real)
    λ = getArgs(t, d)
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1] == 0 ? log.(y) : expm1.(log.(y) .* λ[1]) ./ λ[1]
    return out
end
function evaluate(t::BoxCox, λ, y::Real)
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1] == 0 ? log.(y) : expm1.(log.(y) .* λ[1]) ./ λ[1]
    return out
end



function evaluate_derivative_x(t::BoxCox, y::Real)
    λ = t.λ
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1]==0 ? float(y).^(-1) : float(y).^(λ[1] .-1)
    return out
end
function evaluate_derivative_x(t::BoxCox, d::Dict, y::Real)
    λ = getArgs(t, d)
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1]==0 ? float(y).^(-1) : float(y).^(λ[1] .-1)
    return out
end
function evaluate_derivative_x(t::BoxCox, λ, y::Real)
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1]==0 ? float(y).^(-1) : float(y).^(λ[1] .-1)
    return out
end



function evaluate_derivative_x2(t::BoxCox, d::Dict, y::Real) 
    λ = getArgs(t, d)
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = nothing
    if λ[1]==0
        out = -float(y).^(-2)
    else
        out = (λ[1] .-1) .* float(y).^(λ[1] .-2)
    end
    return out
end



function evaluate_inverse(t::BoxCox, x::Real)
    λ = t.λ
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1]==0 ? Base.exp.(x) : Base.exp.(Base.log.(λ[1].*x.+1)./λ[1])
    return out
end
function evaluate_inverse(t::BoxCox, d::Dict, x::Real)
    λ = getArgs(t, d)
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1]==0 ? Base.exp.(x) : Base.exp.(Base.log.(λ[1].*x.+1)./λ[1])
    return out
end
function evaluate_inverse(t::BoxCox, λ, x::Real)
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    out = λ[1]==0 ? Base.exp.(x) : Base.exp.(Base.log.(λ[1].*x.+1)./λ[1])
    return out
end



function evaluate_derivative_hyperparams(t::BoxCox, d::Dict, y::Real) 
    derivative_dict = Dict()
    λ = getArgs(t, d)
    if "λ" in keys(d) 
        derivative_dict["λ"] = λ[1] == 0 ? 0 : ( ( y.^λ .* log.(y) )./λ - (expm1.(log.(y) .* λ[1]))./(λ^2) ) 
    end
    return derivative_dict
end


function evaluate_derivative_x_hyperparams(t::BoxCox, d::Dict, y::Real)
    derivative_dict = Dict()
    λ = getArgs(t, d)
    if "λ" in keys(d) 
        derivative_dict["λ"] = λ[1] == 0 ? 0 : (  float(y).^(λ[1] .-1) .* log.(y)) 
    end
    return derivative_dict
end
