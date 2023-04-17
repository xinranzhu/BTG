"""
for composed transformation only
sort parameter_names, lower_bound and upper_bound
Inputs:
    parameter_names = ["θ", "var",  "COMPOSED_TRANSFORM_PARAMETER_2", "COMPOSED_TRANSFORM_PARAMETER_1"]
    lower_bound = [[0.00001, 0.], 0.00001,  [0.00001, 0.000001], [0.0, 0.1, 0.2, 0.3]]
    upper_bound = [[10.0, 10.0], 10.0, [20., 20.], [1, 2, 3, 4]];

After processing composed params:
    parameter_names = ["var",  [["a", "b", "c", "d"], ["a", "b"]], "θ"]
    lower_bound = [0.00001, 0.0, 0.1, 0.2, 0.3, 0.00001, 0.000001, [0.00001, 0.]]
    upper_bound = ...

Outputs:
    parameter_names =  ["var",  [["a", "b", "c", "d"], ["a", "b"]], ["θ1", "θ2"]]
    lower_bound = [0.00001, 0, 0.00001,  0.0, 0.1, 0.2, 0.3, 0.00001, 0.000001]
    upper_bound = ...
"""
function parse_input(parameter_names, lower_bound_orig, upper_bound_orig, transform)
    # for single transformation, only parse theta
    lower_bound = deepcopy(lower_bound_orig)
    upper_bound = deepcopy(upper_bound_orig)
    new_parameter_names = convert(Array{Any}, deepcopy(parameter_names))
    if typeof(transform)<:AbstractComposedTransform
        idx_set = []
        for i in 1:length(parameter_names)
            name = parameter_names[i]
            if startswith(name, "COMPOSED_TRANSFORM_PARAMETER")
                push!(idx_set, i)
            end
        end

        composed_parameters = parameter_names[idx_set]
        composed_lower = lower_bound[idx_set]
        composed_upper = upper_bound[idx_set]
        permutation = sortperm(composed_parameters, by = x -> parse(Int64, x[30:end]))

        composed_parameters = composed_parameters[permutation]
        composed_lower = composed_lower[permutation]
        composed_upper = composed_upper[permutation]
        deleteat!(parameter_names, idx_set)
        deleteat!(lower_bound, idx_set)
        deleteat!(upper_bound, idx_set)
        # parameter_names = convert(Array{Union{String, Array{String}, Array{Array{String}}}}, parameter_names)
        # lower_bound = convert(Array{Union{Float64, Array{Float64}, Array{Array{Float64}}} }, lower_bound)
        # upper_bound = convert(Array{Union{Float64, Array{Float64}, Array{Array{Float64}}} }, upper_bound)
        parameter_names = convert(Array{Any}, parameter_names)
        lower_bound = convert(Array{Any}, lower_bound)
        upper_bound = convert(Array{Any}, upper_bound)

        parameter_names = cat(parameter_names, composed_parameters; dims=1)
        composed_lower_flattened = convert(Array{Float64,1}, flatten_nested_list(composed_lower))
        composed_upper_flattened = convert(Array{Float64,1}, flatten_nested_list(composed_upper))

        for i in 1:length(composed_lower_flattened)
            push!(lower_bound, composed_lower_flattened[i])
            push!(upper_bound, composed_upper_flattened[i])
        end

        transform_list = transform.transform_list
        #construct nested parameter names struct
        new_parameter_names = []
        composed_parameter = Array{Any, 1}(undef, length(transform_list))
        for item in parameter_names
            if length(item)>=30
                num = parse(Int64, item[30:end])
                cur_names = transform_list[num].names
                composed_parameter[num] = cur_names
                # push!(new_parameter_names, cur_names)
            else
                push!(new_parameter_names, item)
            end   #determine if normal or composed parameter
        end

        for i in 1:length(transform_list)
            if !isassigned(composed_parameter, i)
                composed_parameter[i] = []
            end
        end
        push!(new_parameter_names, composed_parameter)
    else
        # do nothing for single transformation
    end

    # flatten range for multidimensional theta
    idx_θ = findall(x->x=="θ", new_parameter_names)[1]

    lower_bound_θ = lower_bound[idx_θ]
    upper_bound_θ = upper_bound[idx_θ]
    dim_θ = length(lower_bound_θ)
    if dim_θ > 1
        deleteat!(new_parameter_names, idx_θ)
        deleteat!(lower_bound, idx_θ)
        deleteat!(upper_bound, idx_θ)
        nested_θ_names = []
        for j in 1:dim_θ
            push!(nested_θ_names, "θ$j")
            try
                push!(lower_bound, lower_bound_θ[j])
            catch e
                lower_bound = convert(Array{Any}, lower_bound)
                push!(lower_bound, lower_bound_θ[j])
            end
            try
                push!(upper_bound, upper_bound_θ[j])
            catch e
                upper_bound = convert(Array{Any}, upper_bound)
                push!(upper_bound, upper_bound_θ[j])
            end
        end
        push!(new_parameter_names, nested_θ_names)
    end
    parameter_names = new_parameter_names
    lower_bound = convert(Array{Float64,1}, lower_bound)
    upper_bound = convert(Array{Float64,1}, upper_bound)
    return parameter_names, lower_bound, upper_bound
end
