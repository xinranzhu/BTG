# quadrature rules
"""
Assignment of numerical values to quadrature types
"""
@enum QuadType begin
    Gaussian = 0
    MonteCarlo = 1
    QuasiMonteCarlo = 2
    SparseGrid = 3 #special case, does not rely on Cartesian product
end

"""
Typically a per parameter quadrature information container. Specifies
    - num_points: number quadrature points for this axis (if applicable)
    - quad_type: Gaussian (0), MonteCarlo (1), QuasiMonteCarlo (2), SparseGrid (3)
    - range: integration range for this parameter
Additionally stores corresponding
    - parameter
"""
abstract type AbstractQuadInfo end

struct QuadInfo <:AbstractQuadInfo
    parameter::Parameter
    quad_type::QuadType
    num_points::Union{Nothing, Int64}
    levels::Int64
    range::Union{Nothing, Array{T} where T<:Real}
    function QuadInfo(parameter::Parameter; quad_type::QuadType = QuadType(0),
        num_points::Int64 = 10, levels = 5, range::Array{T} where T<:Real = parameter.range)
        @assert size(range) == size(parameter.range)
        @assert minimum(range[:,1] .>= parameter.range[:,1]) "Range for integration should remain in parameter range"
        @assert minimum(range[:,2] .<= parameter.range[:,2]) "Range for integration should remain in parameter range"
        return new(parameter, quad_type, num_points, levels, range)
    end
end

getparameter(QuadInfo::AbstractQuadInfo)::Parameter = QuadInfo.parameter
getquadtype(QuadInfo::AbstractQuadInfo)::QuadType = QuadInfo.quad_type
getnumpoints(QuadInfo::AbstractQuadInfo)::Union{Nothing, Int64} = QuadInfo.num_points
getrange(QuadInfo::AbstractQuadInfo)::Union{Nothing, Array{T} where T<:Real} = QuadInfo.range

"""
Fully defines quadrature problem. Consists of two parts
- parameters: parameters involved, and which define domain of integration
- quad_info_dict: map from parameter to quad_info object
"""
abstract type AbstractQuadratureSpecs end

struct QuadratureSpecs <: AbstractQuadratureSpecs
    parameters::Union{Array{Parameter}, Array{Array{Array{Parameter}}}} #named parameters in integration domain
    quad_info_dict::Dict{String, QuadInfo} #QuadType and #integration nodes
    function QuadratureSpecs(parameters::Array{Parameter},
        quad_info_dict::Dict{String, T} where T<:AbstractQuadInfo)
        #set default quad_info values
        for p in parameters
            if !(p.name in keys(quad_info_dict))
                quad_info_dict[k] = QuadInfo(p)  # default to 10 Gauss quadrature nodes
            end
        end
        @assert Set(keys(quad_info_dict)) == Set([p.name for p in parameters])
        return new(parameters, quad_info_dict)
    end
end

function build_quad_domain(param_list::Array{Parameter}; quadtype::Int=3, num_points::Int=12, levels::Int=4)
    m = Dict{String, T where T<:AbstractQuadInfo}()
    for param in param_list
        name = param.name
        m[name] = QuadInfo(param; quad_type = QuadType(quadtype), levels=levels, num_points=num_points^(param.dimension))
        # m[name] = QuadInfo(param; quad_type = QuadType(quadtype), levels=levels, num_points=num_points)
    end
    qs = QuadratureSpecs(param_list, m)
    qd = QuadratureDomain(qs)
    return qd
end
