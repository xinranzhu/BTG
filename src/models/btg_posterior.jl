"""
Testing results at single testing point
"""
mutable struct btgPosteriorResults
    data::Dict{String, Array}
    function btgPosteriorResults()
        data = Dict(
                    "pdf" => Array{Function, 1}(),
                    "cdf" => Array{Function, 1}(),
                    "samples" => Array{Array, 1}(),
                    )
        new(data)
    end
end


"""
btgPosterior object contains BTG posterior pdf/cdf, mean and variance.
"""
mutable struct btgPosterior
    model::Btg
    method::String
    testingdata::AbstractTestingData
    # pdf::Array{Function, 1}
    # cdf::Array{Function, 1}
    # pdf::Function
    # cdf::Function
    samples::Array{Real, 1}
    # dpdf::Array{Function, 1}
    # median::Array{Real, 1}
    # variance::Array{Real, 1}
    debug_log::Any
    function posterior_single(i::Int, x_i::Array{T,2}, Fx_i::Array{T,2},
                            pdf_bufferized::Function, cdf_bufferized::Function,
                            MLE=false, test_buffer = Buffer(), quantile_bound=nothing,
                            verbose=false, loose_bound=false,
                            quantile_method = "0order", xtol = nothing, ftol = nothing) where T<:Real

        # results_i = btgPosteriorResults() # initialize a results object for the i-th point
        cdf_i(y) = cdf_bufferized(x_i, Fx_i, y, test_buffer)
        pdf_i(y) = pdf_bufferized(x_i, Fx_i, y, test_buffer)
        # actually quantile_bound_i doesn't depend on y
        if verbose
            println("\n (Prediction $i)----------------")
            @show x_i, Fx_i
        end
        quantile_bound_i(quant) = quantile_bound == nothing ? nothing : quantile_bound(x_i, Fx_i, 0.5, quant; lookup=true, test_buffer=test_buffer)
        
        support_i = [1e-4, 2.0]
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
        quantile_bound_stats = MLE ? nothing : quantile_bound_i
        if quantile_bound == nothing
            quantile_bound_stats = nothing
        end
        sample_i, quantile_info_median = quantile(cdf_i, support_i;
            p=rand(),
            quantile_bound=quantile_bound_stats, verbose=verbose, loose_bound=loose_bound,
            quantile_method = quantile_method, xtol = xtol, ftol = ftol)

        # sample from pdf_i
        # println(typeof(cdf_i))
        # println(typeof(pdf_i))

        # append!(results_i.data["pdf"], [pdf_i])
        # append!(results_i.data["cdf"], [cdf_i])
            

        debug_log = nothing

        return pdf_i, cdf_i, sample_i, debug_log
    end

    function btgPosterior(
        x::Array{T,2},
        Fx::Array{T,2},
        btg::Btg;
        MLE::Bool=false, verbose=false,
        scaling = true, loose_bound=false,
        use_quantile_bound=true,
        quantile_method = "0order",
        quantile_bound_type = "convex_hull",
        dominant_weight_threshold = 0.99,
        test_buffers::Union{Nothing, Dict{Int64, AbstractBuffer}} = nothing,
        xtol = nothing,
        ftol = nothing
        ) where T<:Real

        # pred_time_inside_start = Dates.now()

        method = MLE ? "MLE" : "Bayesian"
        testingdata = TestingData(x, Fx; y0_true=nothing)
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

        results = btgPosteriorResults() # initialize result for testing data
        debug_log = []

        # pdf_array = Array{Function, 1}()
        # cdf_array = Array{Function, 1}()
        samples_array = Array{Real, 1}()

        # pred_single_total_time_inside_start = Dates.now()
        for i in 1:n_test
            # pred_single_inside_preprocess_start = Dates.now()
            if mod(i, 500) == 0
                println("\nPrediction i = $(i)")
            end
            x_i = reshape(x[i, :], 1, dimx)
            Fx_i = reshape(Fx[i, :], 1, dimFx)
            y_i_true = nothing 
            # @show x_i, y_i_true
            cur_test_buffer = test_buffers == nothing ? Buffer() : test_buffers[i]
            # quantile_bound_bufferized(x, Fx, y, quant, test_buffer) =  quantile_bound(x, Fx, y, quant; lookup = true, test_buffer = test_buffer)
            pdf_bufferized(x, Fx, y, test_buffer) = pdf_raw(x, Fx, y; lookup = true, test_buffer = test_buffer)
            cdf_bufferized(x, Fx, y, test_buffer) = cdf_raw(x, Fx, y; lookup = true, test_buffer = test_buffer)
            # @timeit to "single point prediction" results_i, debug_log_i = predict_single(i, x_i, Fx_i, pdf_bufferized, cdf_bufferized;

            # pred_single_inside_call_start = Dates.now()
            pdf_i, cdf_i, sample_i, debug_log_i = posterior_single(i, x_i, Fx_i,
                                                    pdf_bufferized, cdf_bufferized;
                                                    MLE=MLE,
                                                    test_buffer = cur_test_buffer,
                                                    quantile_bound=quantile_bound,
                                                    verbose=verbose, loose_bound=loose_bound,
                                                    quantile_method = quantile_method, xtol = xtol,
                                                    ftol = ftol)
            # scale prediction <- prediction*y_std + y_mean
            Sample_i = Sample_i * train_y_std + train_y_mean
            println("Sample_i = $(sample_i)")
            # append!(pdf_array, cdf_i)
            # append!(cdf_array, pdf_i)
            append!(samples_array, sample_i) 
        end
        new(btg, method, testingdata, samples_array, debug_log)
    end
end



