
function init_btg(trainingdata, transform_name)
    modelname = "Btg"
    feature_size = trainingdata.d
    kernel, params, MLE_range_θ, MLE_range_noise_var = init_kernel_params(
        modelname, feature_size
        )
    my_transform, params, transform_list = init_transform_params(
        transform_name, params; param_range=20.)

    qd = build_quad_domain(params; quadtype=3, levels=2) # qd doesn't matter for MLE method.
    model = Btg(trainingdata, qd, modelname, 
        my_transform, kernel,
        verbose=false
    )
    param_info = [params,  MLE_range_θ, MLE_range_noise_var]
    return model, param_info, transform_list
end


function fit_btg(model, trainingdata, param_info, transform_list; 
    randseed=1234, multistart=10,
    initial_guess=nothing, sobal=false,
    quadtype="SparseGrid",
    total_mass=0.99,
    Bayesian_step=0.2,
    Bayesian_prior_scale=20.,
    range_type="Bayesian_scale")

    params = param_info[1]
    MLE_range_θ = param_info[2]
    MLE_range_noise_var = param_info[3]
    modelname = "Btg"

    parameter_names, upper_bound, lower_bound = get_MLE_bound(params)
    btg_optimize!(
        model, deepcopy(parameter_names), 
        deepcopy(lower_bound), deepcopy(upper_bound);
        multistart=multistart, 
        randseed=randseed, 
        initial_guess=initial_guess, 
        sobol=sobal)
    
    MLE_optimize_status = model.MLE_optimize_status
    if !MLE_optimize_status
        @warn "MLE FAILED to fit a model!"
    end

    opt_param_dict = model.likelihood_optimum.first
    kernel, btg_transform, params = get_btg_params(
        opt_param_dict, 
        my_transform, transform_list, 
        MLE_range_θ, MLE_range_noise_var; 
        param_range=20., 
        Bayesian_prior_scale=Bayesian_prior_scale, 
        Bayesian_step=Bayesian_step,
        range_type=range_type)

    quadtype_dict = Dict("SparseGrid"=>3,"QuasiMonteCarlo"=>2,"MonteCarlo"=>1)
    qd = get_btg_quadrature(params, quadtype_dict[quadtype])

    model = Btg(trainingdata, qd, modelname, btg_transform, kernel; 
        lookup=true, 
        num_weights=nothing, 
        total_mass=total_mass,
        verbose=true);

    return model
end

function predict_btg(x0, Fx0, y0_true, model; verbose=true,
    use_quantile_bound=true,
    quantile_method="brent",
    quantile_bound_type="convex_hull",
    xtol=1e-4,
    ftol=1e-6,
    )
    result = btgPrediction(x0, Fx0, model;
        y_true=y0_true, confidence_level=0.95,
        plot_single=false, MLE=false, 
        scaling=true, verbose=false,
        use_quantile_bound=use_quantile_bound, quantile_method=quantile_method, 
        quantile_bound_type=quantile_bound_type, xtol=xtol, ftol=ftol);

    if verbose
        show_results(result);
    end 
    return result
end

function set_Bayesian_range(optval, MLE_range, Bayesian_prior_scale, Bayesian_step; 
    range_type="fine")
    bound = nothing
    bayesian_range = nothing
    if range_type == "fine"
        if abs(optval) < 1e-3
            bound = min(MLE_range[2], max(0.1, Bayesian_prior_scale*optval))
            bayesian_range = optval > 0 ? [0, bound] : [-bound, 0]
        elseif abs(optval) > 10
            upper = min(MLE_range[2], (1+Bayesian_step)*abs(optval))
            lower = max((1-Bayesian_step)*abs(optval)-10, 0)
            bayesian_range = optval > 0 ? [lower, upper] : [-upper, -lower]
        elseif abs(optval) <= 10 && abs(optval) > 2
            bound = min(MLE_range[2], Bayesian_prior_scale*abs(optval)/2)
            bayesian_range = optval > 0 ? [0, bound] : [-bound, 0]
        else
            bound = min(MLE_range[2], Bayesian_prior_scale*abs(optval)*2)
            bayesian_range = optval > 0 ? [0, bound] : [-bound, 0]
        end
        return reshape(bayesian_range, 1, 2)
    elseif range_type == "Bayesian_scale"
        bound = abs(Bayesian_prior_scale*optval)
        if optval > 0
            return reshape([0, bound], 1, 2)
        else
            return reshape([-bound, bound], 1, 2)
        end
    else
        return reshape(MLE_range, 1, 2)
    end
end

function init_kernel_params(modelname, feature_size; param_range=50.)
    kernel = RBFKernelModule()
    MLE_range_θ = repeat([0. param_range], feature_size)
    MLE_range_var = [0. param_range]
    MLE_range_noise_var = [0. param_range]
    θ = Parameter("θ", feature_size, MLE_range_θ)
    var = Parameter("var", 1, MLE_range_var)
    noise_var = Parameter("noise_var", 1, MLE_range_noise_var)
    params = [θ, var, noise_var]

    kernel_dict_btg_fixed = Dict("θ"=>nothing, "var"=>1.0, "noise_var"=>nothing)
    if modelname == "Btg"
        global pramas
        kernel = RBFKernelModule(kernel_dict_btg_fixed)
        params = [θ, noise_var]
    end
    return kernel, params, MLE_range_θ, MLE_range_noise_var
end

function init_transform_params(transform_name, params; param_range=20.)
    transform_name = split(transform_name, "-")
    transform_list = nothing
    if length(transform_name) == 1 # single transform
        global my_transform
        my_transform = eval(Symbol(transform_name[1]))()
        for name in my_transform.names
            range_cur = reshape([0. param_range], 1, 2)
            if startswith(name, "a") || startswith(name, "c")
                range_cur = reshape([-param_range param_range], 1, 2)
            end
            params = push!(params, Parameter(name, 1, range_cur))
        end
    elseif length(transform_name) > 1 # Composed transform
        global my_transform
        transform_list = []
        for i in 1:length(transform_name)
            transform_cur = eval(Symbol(transform_name[i]))()
            names_cur = transform_cur.names
            num_param_cur = length(names_cur)
            range_set = repeat([0. param_range], num_param_cur)
            for i in 1:num_param_cur
                name = names_cur[i]
                if startswith(name, "a") || startswith(name, "c")
                    range_set[i, 1] = -param_range
                end
            end
            cp_param = Parameter("COMPOSED_TRANSFORM_PARAMETER_$i", num_param_cur, range_set)
            params = push!(params, cp_param)
            push!(transform_list, transform_cur)
        end
        my_transform = ComposedTransformation(transform_list)
    end
    return my_transform, params, transform_list
end


function get_MLE_bound(params)
    parameter_names = [param.name for param in params] # string version of params
    lower_bound = convert(Array{Any}, [param.range[:, 1] for param in params])
    upper_bound = convert(Array{Any}, [param.range[:, 2] for param in params])
    for i in 1:length(lower_bound)
        if length(lower_bound[i]) == 1
            lower_bound[i] = lower_bound[i][1]
            upper_bound[i] = upper_bound[i][1]
        end
    end
    return parameter_names, upper_bound, lower_bound
end

function get_btg_params(opt_param_dict, my_transform, transform_list, MLE_range_θ, MLE_range_noise_var; 
    param_range=50., Bayesian_prior_scale=30., Bayesian_step=0.2,
    fully_bayesian=1, range_type="fine")

    # kernel
    kernel_dict_fixed = Dict{String, Union{T where T<:Real, Array{T} where T<:Real, Nothing}}()
    kernel_dict_fixed["var"] = 1.0
    kernel = RBFKernelModule(kernel_dict_fixed)

    bayesian_range_θ = zeros(dimx, 2)
    for i in 1:dimx
        bayesian_range_θ[i, :] = set_Bayesian_range(opt_param_dict["θ"][i], MLE_range_θ[i,:], Bayesian_prior_scale, Bayesian_step;range_type=range_type)
    end
    bayesian_range_noise_var = set_Bayesian_range(opt_param_dict["noise_var"], MLE_range_noise_var, Bayesian_prior_scale, Bayesian_step;range_type=range_type)
    θ = Parameter("θ", dimx, bayesian_range_θ)
    noise_var = Parameter("noise_var", 1, bayesian_range_noise_var)
    params = [noise_var, θ]
    
    # transforms
    transform_list_new = []
    if typeof(my_transform)<:AbstractComposedTransform
        if fully_bayesian == 1
            for i in 1:length(transform_list)
                opt_vals = opt_param_dict["COMPOSED_TRANSFORM_PARAMETER_$i"]
                param_dim = length(opt_vals)
                param_range_temp = zeros(param_dim, 2)
                names_cur = my_transform.transform_list[i].names
                for j in 1:param_dim
                    name = names_cur[j]
                    if startswith(name, "a") || startswith(name, "c")
                        range_cur = [-param_range, param_range]
                    else
                        range_cur = [0, param_range]
                    end
                    param_range_temp[j, :] = set_Bayesian_range(opt_vals[j], range_cur, 
                        Bayesian_prior_scale, Bayesian_step;range_type=range_type)
                end
                param_cur = Parameter("COMPOSED_TRANSFORM_PARAMETER_$i", param_dim, param_range_temp)
                push!(params, param_cur) 
            end
        else
            # set the parameter of composed transform as MLE learned ones
            # global transform
            # global transform_list_new
            transform_list = my_transform.transform_list
            for i in 1:length(transform_list)
                # global transform_list_new
                transform_cur = transform_list[i]
                names_cur = transform_cur.names
                transform_cur_dict = Dict{String, Any}()
                for j in 1:length(names_cur)
                    transform_cur_dict[names_cur[j]] = opt_param_dict["COMPOSED_TRANSFORM_PARAMETER_$i"][j]
                end
                transform_cur = eval(Symbol(transform_name[i]))(transform_cur_dict)
                transform_list_new = push!(transform_list_new, transform_cur)
            end
            # reform the composed transform with fixed parameters
            my_transform = ComposedTransformation(transform_list_new)
        end
    else
        for name in my_transform.names
            # global params
            opt_temp = opt_param_dict[name]
            range_cur = [0, param_range]
            if startswith(name, "a") || startswith(name, "c")
                range_cur = reshape([-param_range param_range], 1, 2)
            end
            range_tmp = set_Bayesian_range(opt_temp, range_cur, Bayesian_prior_scale, Bayesian_step;range_type=range_type)
            trans_param = Parameter(name, 1, range_tmp)
            params = push!(params, trans_param)
        end
    end

    return kernel, my_transform, params
end


function get_btg_quadrature(params, quadtype; num_points=nothing, levels=nothing)
    int_dimension = sum(param.dimension for param in params)

    # quadtype = 2 # QMC
    # num_points_dict = Dict(1=>25, 2=>12, 3=>6, 4=>4, 5=>3)
    num_points_dict = Dict(1=>30, 2=>20, 3=>7, 4=>5, 5=>4)
    if num_points == nothing
        num_points = int_dimension<=5 ? num_points_dict[int_dimension] : 2
    end

    # quadtype = 3 # sparse
    # levels_dict = Dict(1=>25, 2=>7, 3=>5, 4=>4, 5=>4, 6=>4, 7=>3, 8=>3, 9=>3, 10=>3)
    # levels_dict = Dict(1=>25, 2=>8, 3=>6, 4=>5, 5=>5, 6=>5, 7=>4, 8=>4, 9=>4, 10=>4)
    levels_dict = Dict(1=>25, 2=>10, 3=>6, 4=>5, 5=>5, 6=>4, 7=>4, 
                        8=>4, 9=>4, 10=>4, 11=>4, 12=>4, 13=>3)
    if levels == nothing
        levels = int_dimension <= 10 ? levels_dict[int_dimension] : 3
    end
    # quadtype = int_dimension > 5 ? 3 : 2

    qd = build_quad_domain(params; quadtype=quadtype, levels=levels, num_points=num_points)
    return qd
end