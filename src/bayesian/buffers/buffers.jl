import Base: getindex, setindex!, keys
include("../../core/types.jl")
include("../../kernels/kernels.jl")
include("../../core/data.jl")

#%% Abstractcache - smallest unit of storage
abstract type AbstractCache end

#%% dictionary of caches -second smallest unit of storage
abstract type AbstractBuffer end

"""
A buffer should only contain a single type of cache.
"""
struct Buffer<:AbstractBuffer
    cache_map::Dict{Union{Dict, Array{Dict,1}}, AbstractCache}
    # cache_map::Dict{Union{Dict{String, Union{Float64, Array{Float64, 1}, Array{Float64,2}}}, Array{Dict,1}}, AbstractCache} #, Array{Array{Float64, 1}, 1}}
    function Buffer()::AbstractBuffer
        #return new(Dict())
        # return new(Dict{Union{Dict{String, Union{Float64, Array{Float64, 1}, Array{Float64,2}}}, Array{Dict,1}}, AbstractCache}())
        return new(Dict{Union{Dict{String, Any}, Array{Dict,1}}, AbstractCache}())
    end
    function Buffer(cache_map)
        return new(cache_map)
    end
end

#TODO Force type of y to be something like TNumeric, so type inference is easy

function getindex(x::AbstractBuffer, y::Union{Dict, Array{Dict,1}} where T<:Any)
    return x.cache_map[y]
end

function setindex!(x::AbstractBuffer, a::T where T<:AbstractCache, y::Union{Dict, Array{Dict,1}})
    x.cache_map[y] = a
end

function update!(x::AbstractBuffer, k, v)
    x.cache_map[k] = v
end

function keys(x::AbstractBuffer)
    return keys(x.cache_map)
end

#%% dictionary of buffers - largest unit of storage
abstract type AbstractBufferDict end
"""
Stores an assortment of buffers, each of which contains a dictionary of caches
"""
struct BufferDict<:AbstractBufferDict
    buffer_dict::Dict{String, AbstractBuffer}
    function BufferDict(buffer_dict::Dict{String, T} where T<:AbstractBuffer)
        return new(buffer_dict)
    end
    function BufferDict()
        return new(Dict())
    end
end

"""
Enables easy lookups, which are factored through a BufferDict object,
which represents a dictionary of Buffers
"""
function getindex(x::AbstractBufferDict, y::String)
    return x.buffer_dict[y]
end

function setindex!(x::AbstractBufferDict, a::T where T<:AbstractBuffer, y::String)
    x.buffer_dict[y] = a
end

function keys(x::AbstractBufferDict)
    return keys(x.buffer_dict)
end

#%% Custom Caches
include("caches.jl")
include("btg_caches/caches.jl")
