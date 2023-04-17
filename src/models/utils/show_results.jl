function compare_quantile(r1, r2)
    q1 = r1.quantile_bound_set #bound
    q2 = r2.quantile_bound_set
    if length(q1) != length(q2)
        @warn "Length of quantile info not equal!"
    end
    T1 = []
    T2 = []
    for i = 1:length(q1)
        q1_i = q1[i]
        q2_i = q2[i]
        @assert length(q1_i) == 3
        @assert length(q2_i) == 3
        for j in 1:3
            T1 = append!(T1, q1_i[j][4])
            T2 = append!(T2, q2_i[j][4])
        end
    end
    speedup = round.((T2 .- T1)./T2 .* 100; digits=2)
    @show T1, T2, speedup
    println("\n\n\n\nAverage speedup = $(mean(speedup[2:end])), max = $(maximum(speedup))")
    return T1, T2, speedup, mean(speedup), maximum(speedup)
end



function show_results(pred_result::btgPrediction;
                        verbose=false, plot_results=false,
                        figure_title=nothing, save_path=nothing, plot_transform=false, plot_warped_data=false,
                        sensitivity=false, sensitivity_scale=20, givenrange=nothing)

    modelname = pred_result.model.modelname # modelname = "WarpedGP" or "BTG"
    buffer = pred_result.model.buffer
    method = pred_result.method # "MLE" or "Bayesian"
    pred_median = pred_result.median
    pred_CI = pred_result.credible_interval
    pred_abs_err = pred_result.absolute_error
    pred_sq_err = pred_result.squared_error
    pred_nlpd = pred_result.negative_log_pred_density
    pred_time_cost = pred_result.time_cost;


    testingdata = pred_result.testingdata
    x0 = getposition(testingdata)
    y0_true = getlabel(testingdata)
    ntest = getnumpoints(testingdata)

    trainingdata = pred_result.model.trainingdata
    xtrain_scaled = getposition(trainingdata)
    ytrain_scaled = getlabel(trainingdata)
    xtrain = get_original_x(trainingdata)
    ytrain = get_original_y(trainingdata)
    dimx = getdimension(testingdata)
    ntrain = getnumpoints(trainingdata)
    kernel = pred_result.model.kernel
    transform = pred_result.model.transform
 
    method = pred_result.model.likelihood_optimum.first == Dict() ? "Bayesian" : "MLE"

    count_inf = sum([x==Inf for x in pred_nlpd])
    sum_nlpd = 0.0
    count = 0
    for  x in pred_nlpd
        if x < Inf
            sum_nlpd += x
            count += 1
        end
    end
    mean_nlpd = sum_nlpd/count
    
    println("$(modelname) Prediction Results:")
    println("  Number of training data: ", ntrain)
    println("  Number of testing data: ", ntest)
    println("  Mean squared error = $(round(mean(pred_sq_err), digits=4))")
    # println("   Mean absolute error = $(round(mean(pred_abs_err), digits=4))")
    # println("   Mean negative log predictive density = $(round(mean_nlpd, digits=4)).")
    # println("   CI accuracy = $(pred_result.CI_accuracy)")
    # println("   Time cost: \n       $(pred_time_cost)")

    if verbose
        println("Verbose Mode On... ")
        @show pred_median
        @show pred_CI
        @show pred_abs_err
        @show pred_sq_err
        @show pred_nlpd
        @show pred_time_cost
    end

    if plot_results
        if dimx == 1
            xtest = dropdims(x0, dims=2)
            PyPlot.close("all") #close existing windows
            PyPlot.plot(x0, pred_median, label = "median")
            PyPlot.plot(x0, y0_true, label = "true")
            PyPlot.fill_between(xtest, [x for (x, _) in pred_CI], [y for (_, y) in pred_CI], alpha = 0.3, label = "95% confidence interval")
            PyPlot.scatter(xtrain, ytrain, s = 10, c = "k", marker = "*")
            PyPlot.legend(fontsize=8)
            if figure_title != nothing
                PyPlot.title(figure_title, fontsize=10)
            end
            if save_path != nothing
                PyPlot.savefig("$(save_path)_results.pdf")
                println("Figure saved: $(save_path)_results.pdf")
            end
        elseif dimx == 2
            # x = collect(Float16, range(-2,length=100,stop=2));
            # y = collect(Float16, range(sqrt(2),length=100, stop=2));
            # z = (x.*y).-y.-x.+1;
            # surf(x,y,z);
        else
            # no plot
        end
    end

    opt_dict_result = pred_result.model.likelihood_optimum.first
    opt_dict_transform = nothing
    if opt_dict_result != Dict() # stored some optimal parameters

        if typeof(transform)<:ComposedTransformation
            opt_dict_transform = build_input_dict(transform, reformat(opt_dict_result, transform))
            @show opt_dict_transform
        else
            try
                opt_dict_transform = build_input_dict(transform, opt_dict_result)
            catch err
            end
        end

    
        if plot_transform == true
            save_path1 = save_path == nothing ? nothing : "$(save_path)_opt_transform"
            plot(transform, opt_dict_transform; 
                    yrange=[minimum(ytrain), minimum(ytrain)],
                    figure_title="Optimal transform", save_path=save_path1)
        else
            @warn "No optimal parameters provided, can't plot optimal transform!"
        end

        if plot_warped_data == true
            if dimx == 1
                g(y) = evaluate(transform, opt_dict_transform, y)
                xtrain_copy = dropdims(xtrain, dims=2)
                xtest = dropdims(x0, dims=2)
                PyPlot.close("all") #close existing windows
                PyPlot.scatter(xtrain_copy, ytrain, label = "ytrain")
                PyPlot.scatter(xtrain_copy, g.(ytrain), label = "warped ytrain")
                PyPlot.plot(xtest,  y0_true, label = "normalized ytest_true")
                PyPlot.plot(xtest,  g.(y0_true), label = "warped normalized ytest_true")
                PyPlot.legend(fontsize=8)
                PyPlot.title("Warped Data", fontsize=10)
                if save_path != nothing
                    PyPlot.savefig("$(save_path)_warped_data.pdf")
                    println("Figure saved: $(save_path)_warped_data.pdf")
                end
            end
        else
            @warn "No optimal parameters provided, can't plot warped data!"
        end
    end

    likelihood = pred_result.model.likelihood
    param_list = keys(opt_dict_result) # list of strings
    dict_fixed = deepcopy(opt_dict_result)

    function loglikelihood_slice(x, param; ind=nothing) 
        dict_fixed_copy = deepcopy(dict_fixed)
        if ind == nothing
            dict_fixed_copy[param] = x
        else
            dict_fixed_copy[param][ind] = x
        end
        return likelihood(reformat(dict_fixed_copy, transform), trainingdata, transform, kernel, buffer; 
                            log_scale=true, lookup = true)
    end

    function plot_slice(opt_val, param, func)
        # tmp2 = min(sensitivity_scale*opt_val, max(50., 3*opt_val))
        tmp2 = sensitivity_scale*opt_val
        tmp1 = max(1e-5, opt_val/sensitivity_scale)
        range_param = [tmp1, tmp2]
        if givenrange != nothing
            range_param = [tmp1, givenrange]
        end
        if startswith(param, "a") || startswith(param, "c") # negative values
            range_param = [tmp2, -tmp2]
        end
        xgrid = range(range_param[1], stop=range_param[2], length=200)
        PyPlot.close("all") 
        PyPlot.plot(xgrid, func.(xgrid))
        PyPlot.scatter(opt_val, func(opt_val), marker="*", label = "Optimal value",  c = "r")
        PyPlot.legend(fontsize=8)
        # PyPlot.title("Warped Data", fontsize=10)
        if save_path != nothing
            PyPlot.savefig("$(save_path)_loglikelihood_$(param).pdf")
            println("Figure saved: $(save_path)_loglikelihood_$(param).pdf")
        end
    end

    if sensitivity     
        for param in param_list
            opt_val = dict_fixed[param]
            if length(opt_val) == 1
                opt_val = opt_val[1]
            end
            if startswith(param, "COMPOSED_PARAMETER") || length(opt_val) > 1
                if param == "θ" 
                    dimθ = length(opt_val)
                    if dimθ < 10
                        for i in 1:dimθ
                            loglikelihood_slice_fixed0(x) = loglikelihood_slice(x, param; ind=i)
                            plot_slice(opt_val[i], "θ$(i)", loglikelihood_slice_fixed0)
                        end
                    end
                else
                    idx_transform = parse(Int, param[end])
                    @assert typeof(transform)<:AbstractComposedTransform
                    names = transform.transform_list[idx_transform].names
                    for i in 1:length(names)
                        loglikelihood_slice_fixed1(x) = loglikelihood_slice(x, param; ind=i)
                        plot_slice(opt_val[i], names[i], loglikelihood_slice_fixed1)
                    end
                end
            elseif length(opt_val) == 1
                loglikelihood_slice_fixed2(x) = loglikelihood_slice(x, param)
                plot_slice(opt_val, param, loglikelihood_slice_fixed2)
            end
        end
    end
end

function show_results_loocv(squared_err, absolute_err, CI_accuracy, timer, timer_name)

    println("\n\n---- $(modelname) LOOCV Results ($(method)) ----\n")
    println("   Mean squared error = $(round(mean(squared_err), digits=4))")
    println("   Mean absolute error = $(round(mean(absolute_err), digits=4))")
    println("   CI accuracy = $(round(mean(CI_accuracy), digits=4))")
    println("   Time cost (sec) = $(round(TimerOutputs.time(timer[timer_name])/1e9, digits=2))")


end