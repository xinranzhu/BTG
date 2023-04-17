
function create_params(kernel, modelname, transform_name;
                        kernel_param_range=30., 
                        transform_param_range=50.)

    θ = Parameter("θ", feature_size, repeat([0. kernel_param_range], feature_size))
    var = Parameter("var", 1, [0. kernel_param_range])
    noise_var = Parameter("noise_var", 1, [0. kernel_param_range])
    params = [θ, var, noise_var]
    
    kernel_dict_btg_fixed = Dict("θ"=>nothing, "var"=>1.0, "noise_var"=>nothing)
    if modelname == "Btg"
        # global pramas
        kernel = RBFKernelModule(kernel_dict_btg_fixed)
        params = [θ, noise_var]
    end

    if length(transform_name) == 1 # single transform
        # global my_transform
        my_transform = eval(Symbol(transform_name[1]))()
        for name in my_transform.names
            # global params
            range_cur = reshape([0. transform_param_range], 1, 2)
            if startswith(name, "a") || startswith(name, "c")
                range_cur = reshape([-transform_param_range transform_param_range], 1, 2)
            end
            params = push!(params, Parameter(name, 1, range_cur))
        end
    elseif length(transform_name) > 1 # Composed transform
        # global my_transform
        transform_list = []
        for i in 1:length(transform_name)
            # global params
            transform_cur = eval(Symbol(transform_name[i]))()
            names_cur = transform_cur.names
            num_param_cur = length(names_cur)
            range_set = repeat([0. transform_param_range], num_param_cur)
            for i in 1:num_param_cur
                name = names_cur[i]
                if startswith(name, "a") || startswith(name, "c")
                    range_set[i, 1] =  -transform_param_range
                end
            end
            cp_param = Parameter("COMPOSED_TRANSFORM_PARAMETER_$i", num_param_cur, range_set)
            params = push!(params, cp_param)
            push!(transform_list, transform_cur)
        end
        my_transform = ComposedTransformation(transform_list)
    end

    return params, kernel, my_transform
end


function create_param_opt_bounds(params)
    parameter_names = [param.name for param in params] # string version of params
    lower_bound = convert(Array{Any}, [param.range[:, 1] for param in params])
    upper_bound = convert(Array{Any}, [param.range[:, 2] for param in params])
    for i in 1:length(lower_bound)
        if length(lower_bound[i]) == 1
            lower_bound[i] = lower_bound[i][1]
            upper_bound[i] = upper_bound[i][1]
        end
    end
    return lower_bound, upper_bound, parameter_names
end