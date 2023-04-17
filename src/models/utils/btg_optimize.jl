
"""
["COMPOSED_PARAMETER_1", "COMPOSED_PARAMETER_1" ,"theta"]
lower: [[4, 5], [1, 2], [3, ..., 3]] => [["a", "b"], ["a", "b"], ]
upper: [[6, 7], [8, 9], [6, ..., 6]]

OR
["a", "b", "a", "b", "c"]
lower: [4, 5, 1, 2, 3]
upper: [6, 7, 8, 9, 6]

OR
parameter_names = ["θ", "var",  "COMPOSED_TRANSFORM_PARAMETER_2", "COMPOSED_TRANSFORM_PARAMETER_1"]
lower_bound = [[0.00001, 0.], 0.00001,  [0.00001, 0.000001], [0.0, 0.1, 0.2, 0.3]]
upper_bound = [[10.0, 10.0], 10.0, [20., 20.], [1, 2, 3, 4]];

"""
function btg_optimize!(btg::Btg, parameter_names, lower_bound, upper_bound; initial_guess=nothing, multistart=1, randseed=nothing, sobol=true, verbose=false)
    # check if there's multi-dimensional lengthscale
    lower_bound = deepcopy(lower_bound)
    upper_bound = deepcopy(upper_bound)
    parseθ = false
    btg_parameters = btg.domain.quad_specs.parameters
    btg_parameter_names = [param.name for param in btg_parameters]
    if ("θ" in parameter_names) && ("θ" in btg_parameter_names)
        idxθ_btg = findall(x->x=="θ", btg_parameter_names)
        idxθ_input = findall(x->x=="θ", parameter_names)
        dimθ_btg = btg_parameters[idxθ_btg][1].dimension
        if dimθ_btg > 1 # make sure the lower and upper bound for theta match dimθ
            lower_boundθ = lower_bound[idxθ_input][1]
            upper_boundθ = upper_bound[idxθ_input][1]
            @assert length(lower_boundθ) == dimθ_btg
            @assert length(lower_boundθ) == dimθ_btg
            parseθ = true
        end
    end
    #lower_bound = lower_bound
    #upper_bound = upper_bound
    if typeof(btg.transform)<:AbstractComposedTransform || parseθ
        parameter_names, lower_bound, upper_bound = parse_input(parameter_names, lower_bound, upper_bound, btg.transform)
    end
    function negative_log_likelihood_fixed(arr)
        trainingdata0 = btg.trainingdata
        transform = btg.transform
        kernel = btg.kernel
        buffer = btg.buffer
        dict = nested_list_to_dictionary(parameter_names, arr)
            out = -btg.likelihood(dict, trainingdata0, transform, kernel, buffer; log_scale = true)
        return out
    end

    function derivative_negative_log_likelihood_fixed(arr)
        trainingdata0 = btg.trainingdata
        transform = btg.transform
        kernel = btg.kernel
        buffer = btg.buffer

        dict1 = nested_list_to_dictionary(parameter_names, arr)
        res = btg.log_likelihood_derivatives(dict1, trainingdata0, transform, kernel, buffer)
        out = -dictionary_to_flattened_list(parameter_names, res)
        return out
    end

    function derivative_negative_log_likelihood_fixed!(store, input)
        store .= derivative_negative_log_likelihood_fixed(input)
    end

    function generate_initial_guess_sequence(N_sample)
        s = SobolSeq(lower_bound, upper_bound) # length d
        N = hcat([next!(s) for i = 1:N_sample]...)'  # N by d
        return N
    end

    generate_initial_guess(;randseed=nothing) = [rand(MersenneTwister(randseed))*(u-l) + l for (l, u) in zip(lower_bound, upper_bound)]

    function inner_optimize(;initial_guess=nothing)
        init = initial_guess == nothing ? (@warn "No initial guess, generating a random one."; generate_initial_guess()) : initial_guess
        #(r1, r2, r3, r4) = checkDerivative(negative_log_likelihood_fixed, derivative_negative_log_likelihood_fixed, init)
        lower_bound = convert(Array{Float64,1}, lower_bound)
        upper_bound = convert(Array{Float64,1}, upper_bound)
        out = false
        optimal_parameter_vals_min = Inf
        optimal_status = false
        
        try
            res = Optim.optimize(negative_log_likelihood_fixed, derivative_negative_log_likelihood_fixed!,
                                lower_bound, upper_bound, init, Fminbox(LBFGS()), 
                                Optim.Options(show_trace = false))
            optimal_parameter_vals = Dict()
            optimal_parameter_vals_array = res.minimizer
            optimal_parameter_vals_array_new = flattened_list_to_nested_list(parameter_names, optimal_parameter_vals_array)
            # optimal_parameter_vals_array_new  = [1., [[2., 3., 4., 5.], [6., 7.]], [8., 9.]]
            # parameter_names = ["var",  [["a", "b", "c", "d"], ["a", "b"]], ["θ1", "θ2"]]
            for i = 1:length(parameter_names)
                param = parameter_names[i]
                if typeof(param)<:Array && typeof(param[1])<:Array # [["a", "b", "c", "d"], ["a", "b"]]
                    for j in 1:length(param)
                        if length(param[j]) > 0
                            key = "COMPOSED_TRANSFORM_PARAMETER_"*"$j"
                            optimal_parameter_vals[key] = optimal_parameter_vals_array_new[i][j]
                        end
                    end
                elseif typeof(param)<:Array && typeof(param[1])<:String # ["θ1", "θ2"]
                    key = "θ"
                    optimal_parameter_vals[key] = convert(Array{Float64, 1}, optimal_parameter_vals_array_new[i])
                else
                    optimal_parameter_vals[param] = optimal_parameter_vals_array[i]
                end
            end
            if verbose
                print("\nMLE success")
                print("\nMLE opt: initial guess = ", init)
                print("\nMLE opt: results = ", optimal_parameter_vals=>res.minimum)
            end
            optimal_parameter_vals_min = optimal_parameter_vals=>res.minimum
            optimal_status = Optim.converged(res)
            out = true
        catch e
            out = false
            optimal_parameter_vals_min = Inf
            optimal_status = false
        end
        return optimal_parameter_vals_min, optimal_status, out
    end

    if sobol == true
        initial_guess_set = generate_initial_guess_sequence(multistart)
        initial_guess = initial_guess == nothing ? (println(initial_guess_set[1, :]); initial_guess_set[1, :]) : initial_guess
    else
        initial_guess = initial_guess == nothing ? generate_initial_guess(;randseed=randseed) : initial_guess
    end

    minimum_set = []
    cur_status = false
    MLE_optimize_status = false
    optimal_pair = nothing
    out = false

    # println("In btg optimize, randseed=$(randseed).")
    count_iter = 1
    while out == false && count_iter < 100
        optimal_pair, MLE_optimize_status, out = inner_optimize(;initial_guess=initial_guess)
        if out == false
            print("\nFailed opt, changing randseed to $(randseed+1)\n")
            randseed += 1
            initial_guess = generate_initial_guess(;randseed=randseed)
        end
        count_iter += 1
    end
    push!(minimum_set, optimal_pair)
    count = 1

    count_iter = 1
    while multistart > 1 && count < multistart
        count += 1
        # @info "multistart round $(count)"
        if randseed != nothing
            randseed += count*10
        end
        inner_initial_guess = sobol == true ? initial_guess_set[count, :] : generate_initial_guess(;randseed=randseed)
        if verbose==true
            print("Multistart Iteration: $(count)\n")
            @show inner_initial_guess
        end
        while out == false && count_iter < 100
            optimal_pair, cur_status, out = inner_optimize(;initial_guess=inner_initial_guess)
            if out == false
                print("\nFailed opt, changing randseed to $(randseed+1)\n")
                randseed += 1
                inner_initial_guess = generate_initial_guess(;randseed=randseed)
            end
            count_iter += 1
        end
        push!(minimum_set, optimal_pair)
        MLE_optimize_status = max(MLE_optimize_status, cur_status)
    end
    # select the minimum of minimum
    idx = argmin([val.second for val in minimum_set])
    btg.likelihood_optimum = minimum_set[idx]
    btg.MLE_optimize_status = MLE_optimize_status
    return minimum_set[idx]
end
