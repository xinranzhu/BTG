"""
Testing results at single testing point
"""
mutable struct btgResults
    time::Dict{String, T} where T<:Real
    data::Dict{String, Array}
    function btgResults()
        time = Dict(
                    "time_median" => 0.,
                    "time_CI" => 0.,
                    "time_eval" => 0.,
                    "time_preprocess" => 0.,
                    "time_total" => 0.
                    )
        data = Dict(
                    "pdf" => Array{Function, 1}(),
                    "cdf" => Array{Function, 1}(),
                    # "dpdf" => Array{Function, 1}(),
                    "median" => Array{Real, 1}(),
                    "credible_interval" => Array{Array{Real}, 1}(),
                    "absolute_error"=> Array{Real, 1}(),
                    "squared_error"=> Array{Real, 1}(),
                    "negative_log_predictve_density"=> Array{Real, 1}(),
                    "quantile_info"=> Array{Array, 1}()
                    )
        new(time, data)
    end
end


"""
btgPredict object contains prediction results, including
"""
mutable struct btgPrediction
    model::Btg
    method::String
    testingdata::AbstractTestingData
    pdf::Array{Function, 1}
    cdf::Array{Function, 1}
    # dpdf::Array{Function, 1}
    median::Array{Real, 1}
    credible_interval::Array{Array, 1}
    absolute_error::Union{Array{T, 1}, Nothing} where T<:Real
    squared_error::Union{Array{T, 1}, Nothing} where T<:Real
    negative_log_pred_density::Union{Array{T, 1}, Nothing} where T<:Real
    CI_accuracy::Union{Nothing, T} where T<:Real
    time_cost::Dict{String, T} where T<:Real
    confidence_level::T where T<:Real
    quantile_bound_set::Any
    debug_log::Any
    function predict_single(i::Int, x_i::Array{T,2}, Fx_i::Array{T,2},
                            pdf_bufferized::Function, cdf_bufferized::Function;
                            y_i_true::T = nothing, confidence_level = .95, 
                            train_y_mean::T = nothing, train_y_std::T = nothing,
                            plot::Bool=false, figureid::Union{String, Nothing}=nothing,
                            testid::Union{String, Nothing}=nothing,
                            MLE=false, test_buffer = Buffer(), quantile_bound=nothing,
                            verbose=false, loose_bound=false,
                            quantile_method = "0order", xtol = nothing, ftol = nothing)::BtgResult where T<:Real

        # pred_single_inside_start = Dates.now()
        results_i = btgResults() # initialize a results object for the i-th point
        to = TimerOutput()
        @timeit to "time_total" begin
            @timeit to "time_preprocess" begin
                cdf_i(y) = cdf_bufferized(x_i, Fx_i, y, test_buffer)
                pdf_i(y) = pdf_bufferized(x_i, Fx_i, y, test_buffer)
                if verbose
                    println("\n (Prediction $i)----------------")
                    @show x_i, Fx_i
                end
                quantile_bound_i(quant) = quantile_bound == nothing ? nothing : quantile_bound(x_i, Fx_i, 0.5, quant; lookup=true, test_buffer=test_buffer)

                support_i = [1e-4, 5.0]
                if !MLE
                    function pdf_built_in_test_buffer(x0, Fx0, y0; test_buffer = test_buffer)
                        pdf_bufferized(x0, Fx0, y0, test_buffer)
                    end
                    function cdf_built_in_test_buffer(x0, Fx0, y0; test_buffer = test_buffer)
                        cdf_bufferized(x0, Fx0, y0, test_buffer)
                    end
                    support_i, int_i = pre_process(x_i, Fx_i,
                        pdf_built_in_test_buffer, cdf_built_in_test_buffer, test_buffer; verbose=verbose)
                end
            end
            quantile_bound_stats = MLE ? nothing : quantile_bound_i
            if quantile_bound == nothing
                quantile_bound_stats = nothing
            end
            @timeit to "time_median" begin
                global median_i
                global quantile_info_median
                median_i, quantile_info_median = quantile(cdf_i, support_i;
                    quantile_bound=quantile_bound_stats, verbose=verbose, loose_bound=loose_bound,
                    quantile_method = quantile_method, xtol = xtol, ftol = ftol)
                median_i = median_i .* train_y_std .+ train_y_mean
            end
            @timeit to "time_CI" begin
                global CI_i
                global quantile_info_CI
                CI_i, quantile_info_CI = credible_interval(cdf_i, support_i;
                mode=:equal, wp=confidence_level, quantile_bound=quantile_bound_stats,
                    verbose=verbose, loose_bound=loose_bound, quantile_method = quantile_method, xtol = xtol, ftol = ftol)
                CI_i = CI_i .* train_y_std .+ train_y_mean
            end
            # quantile_info_median = [p1, [O1, O2], p1*, T1]
            # quantile_info_CI = [[p2, [O1, O2], p2*, T2], [[p3, [O1, O2], p3*, T3]]]
            quantile_info = []
            push!(quantile_info, quantile_info_median)
            push!(quantile_info, quantile_info_CI[1][1])
            push!(quantile_info, quantile_info_CI[1][2])
            append!(results_i.data["pdf"], [pdf_i])
            append!(results_i.data["cdf"], [cdf_i])
            # append!(results_i.data["dpdf"], [dpdf_i])
            append!(results_i.data["median"], [median_i])
            append!(results_i.data["credible_interval"], [CI_i])
            append!(results_i.data["quantile_info"], [quantile_info])
            if y_i_true != nothing
                @timeit to "time_eval" begin
                    append!(results_i.data["absolute_error"], [abs(y_i_true - median_i)])
                    append!(results_i.data["squared_error"], [(y_i_true - median_i)^2])
                    nlpd_i = nothing
                    try
                        nlpd_i = -log(pdf_i((y_i_true-train_y_mean)/train_y_std))
                    catch e
                        nlpd_i = 30
                        @warn "Negative pdf value", pdf_i((y_i_true-train_y_mean)/train_y_std)
                        # nlpd_i = -log(pdf_i(y_i_true)) definition of nlpd
                    end
                    append!(results_i.data["negative_log_predictve_density"], [nlpd_i])
                end
            end
        end
        if plot
            y_i_true_scaled = (y_i_true - train_y_mean)/train_y_std
            # y0_grid = y_i_true_scaled*0.9:0.001:y_i_true_scaled*1.1
            y0_grid = range(1e-5, stop=1.1, length=200)
            pdf_grid = pdf_i.(y0_grid)
            cdf_grid = cdf_i.(y0_grid)
            PyPlot.clf()
            PyPlot.plot(y0_grid, pdf_grid)
            if y_i_true != nothing
                PyPlot.vlines(y_i_true_scaled, 0, pdf_i(y_i_true_scaled), label = "true value",  colors = "r")
            end
            PyPlot.title("Posterior pdf at x0 = $(x_i), y=$(y_i_true_scaled)")
            PyPlot.legend(fontsize=8)
            PyPlot.grid()
            PyPlot.savefig("Test$(testid)_posterior_pdf_$(figureid).pdf")
        end

        results_i.time["time_preprocess"] += round(TimerOutputs.time(to["time_total"]["time_preprocess"])/1e9, digits=2)
        results_i.time["time_median"] += round(TimerOutputs.time(to["time_total"]["time_median"])/1e9, digits=2)
        results_i.time["time_CI"] += round(TimerOutputs.time(to["time_total"]["time_CI"])/1e9, digits=2)
        results_i.time["time_eval"] += round(TimerOutputs.time(to["time_total"]["time_eval"])/1e9, digits=2)
        results_i.time["time_total"] += round(TimerOutputs.time(to["time_total"])/1e9, digits=2)
        debug_log = nothing

        # pred_single_inside_end = Dates.now()
        # pred_single_inside =  round(((pred_single_inside_end - pred_single_inside_start) / Millisecond(1000)), digits=2)
        # print("\npred_single_inside = $(pred_single_inside)s")
        return results_i, debug_log
    end

    function btgPrediction(
        x::Array{T,2},
        Fx::Array{T,2},
        btg::Btg;
        y_true::Union{Array{T}, Nothing} = nothing,
        confidence_level::T = .95,
        plot_single::Bool=false,
        testid::Union{String, Nothing}=nothing,
        MLE::Bool=false, verbose=false,
        scaling = true, loose_bound=false,
        use_quantile_bound=true,
        quantile_method = "0order",
        quantile_bound_type = "convex_hull",
        dominant_weight_threshold = 0.99,
        test_buffers::Union{Nothing, Dict{Int64, AbstractBuffer}} = nothing,
        xtol = nothing,
        ftol = nothing,
        ) where T<:Real

        # pred_time_inside_start = Dates.now()

        method = MLE ? "MLE" : "Bayesian"
        testingdata = TestingData(x, Fx; y0_true = y_true)
        train_x_mean = btg.trainingdata.x_mean
        train_x_std = btg.trainingdata.x_std
        train_y_mean = btg.trainingdata.y_mean
        train_y_std = btg.trainingdata.y_std
        # scale test_x <- (test_x - mean)/std
        x = broadcast(-, x, train_x_mean)
        x = broadcast(/, x, train_x_std)
        n_test = getnumpoints(testingdata)
        dimx = getdimension(testingdata)
        dimFx = getcovariatedimension(testingdata)
        pdf_raw = btg.posterior_pdf
        cdf_raw = btg.posterior_cdf
        modelname = btg.modelname
        conditional_posterior = eval(Symbol("$(modelname)ConditionalPosterior"))
        quantile_bound = nothing
        if use_quantile_bound == true
            if quantile_bound_type == "convex_hull"
                quantile_bound = btg.quantile_bound
            elseif quantile_bound_type == "weight_bound"
                quantile_bound = btg.quantile_bound_single_weight
            elseif quantile_bound_type == "dominant_weights"
                quantile_bound = btg.quantile_bound_dominant_weights(dominant_weight_threshold)
            elseif quantile_bound_type == "composite"
                quantile_bound = btg.quantile_bound_composite
            else
                error("Quantile bound not defined. Your input is: $quantile_bound")
            end
             # quantile_bound()
        end
        if MLE #don't use btg.posterior_cdf or btg.posterior_cdf, instead fix optimal hyperparameters
            likelihood_optimum = btg.likelihood_optimum
            parameters_optimum = likelihood_optimum.first
            print("\nUsing MLE optimal parameters in prediction $(parameters_optimum)")
            pdf_raw(x, Fx, y; lookup = true, test_buffer = Buffer()) = conditional_posterior(parameters_optimum, x, Fx, y,
                                        btg.trainingdata, btg.transform, btg.kernel, btg.buffer;
                                        lookup = lookup, test_buffer = test_buffer, scaling = scaling)[1]
            cdf_raw(x, Fx, y; lookup = true, test_buffer = Buffer()) = conditional_posterior(parameters_optimum, x, Fx, y,
                                        btg.trainingdata, btg.transform, btg.kernel, btg.buffer;
                                        lookup = lookup, test_buffer = test_buffer, scaling = scaling)[2]
        end

        results = btgResults() # initialize result for testing data
        debug_log = []
        # predict one by one, could put in parallel

        # pred_single_total_time_inside_start = Dates.now()
        for i in 1:n_test
            # pred_single_inside_preprocess_start = Dates.now()
            if mod(i, 500) == 0
                println("\nPrediction i = $(i)")
            end
            x_i = reshape(x[i, :], 1, dimx)
            Fx_i = reshape(Fx[i, :], 1, dimFx)
            y_i_true = y_true == nothing ? nothing : y_true[i]
            # cur_test_buffer = test_buffers == nothing ? Buffer() : test_buffers[i]
            cur_test_buffer=Buffer()
            pdf_bufferized(x, Fx, y, test_buffer) = pdf_raw(x, Fx, y; lookup = true, test_buffer = test_buffer)
            cdf_bufferized(x, Fx, y, test_buffer) = cdf_raw(x, Fx, y; lookup = true, test_buffer = test_buffer)

            results_i, debug_log_i = predict_single(i, x_i, Fx_i, pdf_bufferized, cdf_bufferized;
                                    y_i_true = y_i_true, confidence_level = confidence_level,
                                    plot=plot_single, figureid="$(i)", testid=testid, MLE=MLE,
                                    test_buffer = cur_test_buffer, quantile_bound=quantile_bound,
                                    train_y_mean = train_y_mean, train_y_std = train_y_std,
                                    verbose=verbose, loose_bound=loose_bound,
                                    quantile_method = quantile_method, xtol = xtol,
                                    ftol = ftol)

            merge_results!(results, results_i)
            
        end
        
        pdf = results.data["pdf"]
        cdf = results.data["cdf"]
        # dpdf = results.data["dpdf"]
        median = results.data["median"]
        credible_interval = results.data["credible_interval"]
        quantile_bound_set = results.data["quantile_info"]
        time_cost = results.time
        absolute_error = nothing
        squared_error = nothing
        negative_log_pred_density = nothing
        CI_accuracy = nothing
        if y_true != nothing # evaluate results
            absolute_error = results.data["absolute_error"]
            squared_error = results.data["squared_error"]
            negative_log_pred_density = results.data["negative_log_predictve_density"]
            CI_accuracy = sum( [ (y_true[i] >= credible_interval[i][1])*(y_true[i] <= credible_interval[i][2]) for i in 1:length(y_true) ]) /n_test
        end

        new(btg, method, testingdata, pdf, cdf, median, credible_interval, absolute_error,
            squared_error, negative_log_pred_density, CI_accuracy,
            time_cost, confidence_level, quantile_bound_set, debug_log)
    end
end

"""
Input testing points, covariates, and quadrature nodes. Assumes covariance
buffers are already defined in btg object, i.e. Cholesky decompositions of
Kernel matrices are already computed.
"""
function btgBuildTestBuffers(
    x0::Array{T,2},
    Fx0::Array{T,2},
    btg::Btg;
    )::Dict{Int64, AbstractBuffer} where T<:Real
    buffer_dict = btg.buffer
    kernel_module_default = btg.kernel
    train = btg.trainingdata
    covariance_buffer_name = "train_covariance_buffer" #TODO turn this into a type, for more security and stability?
    n = getnumpoints(train) #number traiing poitns
    x = getposition(train)
    m = size(x0, 1) #number of testing points
    #construct test cache and place in buffer
    test_buffers = Dict(i => Buffer() for i = 1:m)
    iter = btg.iter_normalized
    for (node_dict, w) in iter #loop over quadrature nodes
        Bθ_mat = zeros(n, m)
        covariance_dict = build_input_dict(kernel_module_default, node_dict) #TODO can we replace d with n?
        covariance_matrix_cache = lookup_or_compute(buffer_dict, covariance_buffer_name,
            CovarianceMatrixCache(), covariance_dict, train, kernel_module_default)
        kernel_module = covariance_matrix_cache.kernel_module
        for k = 1:m #construct all train-test covariance vectors
            x0_cur = x0[k:k, :]
            Bθ = reshape(KernelMatrix(kernel_module, x0_cur, x, noise_var = 0.0), n, 1)
            Bθ_mat[:, k:k] = Bθ
        end
        cholKθ = covariance_matrix_cache.chol #lookup precomputed cholKθ
        Kθ_inv_Bθ_mat = cholKθ\Bθ_mat # THE ASYMPTOTICALLY MOST EXPENSIVE OPERATION
        for k = 1:m #construct part of kth test buffer
            Kθ_inv_B = Kθ_inv_Bθ_mat[:, k:k]
            Bθ = Bθ_mat[:, k:k]
            x0_cur = x0[k:k, :]
            Eθ = KernelMatrix(kernel_module, x0_cur; noise_var = 0.0, jitter = 1e-10)
            test_key_dict = merge(covariance_dict, Dict("x0"=> x0_cur))
            remove_brackets!(test_key_dict)
            Dθ = Eθ - Bθ' * Kθ_inv_B .+ covariance_dict["noise_var"][1] .+  1e-10
            update!(test_buffers[k], test_key_dict, TestingCache(reshape(Bθ, 1, n), Eθ, Dθ)) #Bθ should be row vector
        end
    end
    return test_buffers
end
