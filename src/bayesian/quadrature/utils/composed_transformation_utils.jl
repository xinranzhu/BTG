"""
String name based reformatter. Takes parameters with string names
and reformats them into a dictionary containing a nested array of
dicts of name-value mappings. Used for evaluating likelihood.

Convention: "COMPOSED_TRANSFORM_PARAMETER_i"
Input:
d is from the iterator, for example
d = Dict("var"=>[1.0], "COMPOSED_TRANSFORM_PARAMETER_2"=>[0.1, 0.2])
But there are 2 components in the composed transform, params of transform 1 are fixed

Output:
new_d = Dict("var"=>[1.0], "COMPOSED_TRANSFORM_DICTS"=>[Dict(), Dict("a"=>1, "b"=>2)])
We keep slots for every component to keep the order for build_input_dict later (complete)
"""
function reformat(d::Dict, transform::AbstractTransform)
    if !(typeof(transform)<:AbstractComposedTransform)
        return d
    end
    d = deepcopy(d)
    transform_list = transform.transform_list
    array_dicts = Array{Dict, 1}(undef, length(transform_list))
    for k in keys(d)
        @assert typeof(k)<: String
        if startswith(k, "COMPOSED")
            if !(startswith(k, "COMPOSED_TRANSFORM_PARAMETER"))
                @warn "It is possible a composed transform parameter was not correctly named.
                Must have the form COMPOSED_TRANSFORM_PARAMETER_i, where i indicates the transform
                it belongs to. Your name was $k"
            end
        end
        if length(k) >= 30
            if k[1:28] == "COMPOSED_TRANSFORM_PARAMETER"
                num = parse(Int64, k[30:end])
                names = transform_list[num].names
                array_dicts[num] = Dict(names[i] => d[k][i] for i = 1:length(names))
                delete!(d, k)
            end
        end
    end
    # initialized = false
    for i = 1:length(transform_list)
        if isassigned(array_dicts, i) == false
            array_dicts[i] = Dict()
        end
    end
    #TODO find beter solution to type issue: cannot set element of Dict{String, Float64} to be Array{Dict}
    # new_d = Dict{Any, Union{T, G} where T<:Real where G<:Array{W} where W<:Dict}()
    new_d = Dict{Any, Any}()
    new_d["COMPOSED_TRANSFORM_DICTS"] = array_dicts
    for item in keys(d)
        new_d[item] = d[item]
    end
    return new_d
end
