abstract type AbstractTransform end
abstract type NonlinearTransform <: AbstractTransform end
abstract type AbstractComposedTransform <: AbstractTransform end

# using PyPlot


"""
Faster/optimized build_input_dict
"""
function build_input_dict(t::AbstractTransform, d::Dict)::Dict
    list = t.names
    transform_dict = deepcopy(t.PARAMETER_DICT) # 6 microseconds?
    dkeys = keys(d)
    for item in list
        if item in dkeys
            cur = d[item]
            if typeof(cur)<:Array && length(cur) == 1
                transform_dict[item] = cur[1]
            else
                transform_dict[item] = cur
            end
        end
    end
    if false #can't afford to check user input every single time...just make sure supplied info is enough...
        cond = true
        for k in keys(transform_dict)
            if transform_dict[k]==nothing
                cond = false
            end
        end
        if cond == false || !(issubset(keys(transform_dict), list) && issubset(list, keys(transform_dict))) #swap order of these
            error("Supplied information from d, combined with default values from t, do not
                  constitute complete set of info for building full input dict")
        end
    end
    return transform_dict
end


"""
Builds complete input dict for transformation using default values
    stored in AbstractTransform object and user-supplied values from d
"""
function build_input_dict_original(t::AbstractTransform, d::Dict)
    list = t.names
    transform_dict = deepcopy(t.PARAMETER_DICT) # 6 microseconds?
    for item in list
        if item in keys(d)
            if typeof(d[item])<:Array && length(d[item]) == 1
                transform_dict[item] = d[item][1]
            else
                transform_dict[item] = d[item]
            end
        end
    end
    cond = true
    for k in keys(transform_dict)
        if transform_dict[k]==nothing
            cond = false
        end
    end
    if !(issubset(keys(transform_dict), list) && issubset(list, keys(transform_dict))) || cond == false
        error("Supplied information from d, combined with default values from t, do not
              constitute complete set of info for building full input dict")
    end
    return transform_dict
end


# function plot(t::AbstractTransform, d::Dict;
#                 yrange=[0,100], figure_title=nothing, save_path=nothing)

#     @assert length(yrange) == 2

#     ygrid = collect(yrange[1]:0.01:yrange[2])
#     g(y) = evaluate(t, d, y)
#     gygrid = g.(ygrid)

#     PyPlot.close("all") #close existing windows
#     PyPlot.plot(ygrid, gygrid)

#     if figure_title != nothing
#         PyPlot.title(figure_title, fontsize=10)
#     end
#     if save_path != nothing
#         PyPlot.savefig("$(save_path).pdf")
#         println("Figure saved: $(save_path).pdf")
#     end

# end

getParameterDict(t::AbstractTransform)::Dict = t.PARAMETER_DICT

"""
Here we use push instead of pre-allocating space for args_list, which is highly inefficient.
"""
function getArgs_original(t::AbstractTransform, dict::Dict)
    args_list = []
    param_names = t.names
    num_names = length(param_names)
    for name in param_names
        if isdefined(t, num_names+1) #check PARAMETER_DICTIONARY is defined
            if name in keys(dict)
                push!(args_list, dict[name])
            elseif name in keys(t.PARAMETER_DICT)
                push!(args_list, t.PARAMETER_DICT[name])
            else
                error("$(name) not supplied or defined")
            end
        else
            @assert Set(keys(dict)) == Set(param_names)
            push!(args_list, dict[name])
        end
    end
    for item in args_list
        item = length(item) == 1 ? item[1] : item
    end
    if length(args_list) == 1
        args_list = args_list[1]
    end
    return args_list
end

"""
Here we pre-allocate space for args_list
"""
function getArgs(t::AbstractTransform, dict::Dict)::Union{Float64, Array{Float64}}
    param_names = t.names
    num_names = length(param_names)
    args_list = Array{Float64, 1}(undef, length(param_names))
    for i = 1:num_names
        cur_name = param_names[i]
        if isdefined(t, num_names+1) #check if t.parameter_dict is defined
            if cur_name in keys(dict)
                args_list[i] = dict[cur_name]
            elseif cur_name in keys(t.PARAMETER_DICT)
                args_list[i] = t.PARAMETER_DICT[cur_name]
            else
                error("$(name) not supplied or defined")
            end
        else
            @assert Set(keys(dict)) == Set(param_names)
            args_list[i] = dict[cur_name]
        end
    end
    for item in args_list
        item = length(item) == 1 ? item[1] : item
    end
    if num_names == 1
        args_list = args_list[1]
    end
    return args_list
end
