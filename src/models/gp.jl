mutable struct GpPrediction
    trainingdata::TrainingData
    testingdata::AbstractTestingData
    model::GPE # the GP model from GaussianProcesses
    likelihood_optimum::Pair{A, T} where A<:Dict where T<:Float64
    res::W where W<:Optim.MultivariateOptimizationResults
    mean::Array{T} where T<:Real # posterior mean, length=ntest
    stdv::Array{T} where T<:Real # posterior standard deviation, length=ntest
    absolute_error::Union{Array{T, 1}, Nothing} where T<:Real
    squared_error::Union{Array{T, 1}, Nothing} where T<:Real
    negative_log_pred_density::Union{Array{T, 1}, Nothing} where T<:Real
    time_cost::Dict{String, T} where T<:Real
    confidence_level::T where T<:Real
    CI_accuracy::Union{Nothing, T} where T<:Real
    function GpPrediction(trainingdata::TrainingData, x::Array{T,2}, Fx::Array{T,2}, model::GPE;
                          y_true::Union{Array{T}, Nothing} = nothing, confidence_level::T = .95,
                          expid::Union{String, Nothing}=nothing) where T<:Real
        
        res = optimize!(gpmodel)

        if !Optim.converged(res) 
            @warn "In Gaussian Processes MLE, optimizer fails to converge."
        end

        # res.minimizer # [log(noise_var)/2, log(l), log(var)]
        # theta here is an array, others are scalars
        likelihood_optimum = Dict("θ" => gpmodel.kernel.iℓ2,
                                  "var" => gpmodel.kernel.σ2,
                                  "noise_var" => exp(2*res.minimizer[1])
                                  ) => res.minimum

        if confidence_level != 0.95
            @warn "Not doing 0.95 confidence interval!"
        end
        testingdata = TestingData(x, Fx; y0_true = y_true)
        train_x_mean = trainingdata.x_mean
        train_x_std = trainingdata.x_std
        train_y_mean = trainingdata.y_mean
        train_y_std = trainingdata.y_std
        # scale test_x <- (test_x - mean)/std
        x = broadcast(-, x, train_x_mean)
        x = broadcast(/, x, train_x_std)
        ntest = getnumpoints(testingdata)
        dimx = getdimension(testingdata)
        dimFx = getcovariatedimension(testingdata)
        xtest = reshape(x, dimx, ntest)
        
        to = TimerOutput()
        time_cost = Dict("time_total" => 0.)
        absolute_error = nothing
        squared_error = nothing
        negative_log_pred_density = nothing
        CI_accuracy = nothing
        @timeit to "time_total" begin
            μ, σ² = predict_y(model, xtest)
            μ = reshape(μ, ntest)
            stdv = reshape((sqrt.(σ²)), ntest)
            μ = μ .* train_y_std .+ train_y_mean
            stdv = stdv .* train_y_std .+ train_y_mean

            if y_true != nothing
                z = 1.96 
                if confidence_level != 0.95
                    @warn "confidence_level != 0.95, need to change z value"
                end
                error = reshape(y_true .- μ, ntest)
                absolute_error = abs.(error)
                squared_error = absolute_error.^2
                negative_log_pred_density =  -(log.(Distributions.pdf.(Normal(), error./stdv)./stdv))
                CI_accuracy = sum( ( y_true.>= (μ.-z.*stdv) ) .* ( y_true .<= (μ.+z.*stdv) )  ) / ntest
            end
        end
        time_cost["time_total"] = TimerOutputs.time(to["time_total"])/1e9

        return new(trainingdata, testingdata, model, likelihood_optimum, res,
                    μ, stdv, absolute_error, squared_error, negative_log_pred_density, 
                    time_cost, confidence_level, CI_accuracy)
    end
end

function show_results(pred_result::GpPrediction; 
                      verbose=false, plot_results=false, 
                      figure_title=nothing, save_path=nothing)

    pred_mean = pred_result.mean
    pred_stdv = pred_result.stdv
    pred_abs_err = pred_result.absolute_error
    pred_sq_err = pred_result.squared_error
    pred_nlpd = pred_result.negative_log_pred_density
    pred_time_cost = pred_result.time_cost
    confidence_level = pred_result.confidence_level

    testingdata = pred_result.testingdata
    x0 = getposition(testingdata)
    y0_true = getlabel(testingdata)
    ntest = getnumpoints(testingdata)

    trainingdata = pred_result.trainingdata
    xtrain_scaled = getposition(trainingdata)
    ytrain_scaled = getlabel(trainingdata)
    ntrain = getnumpoints(trainingdata)
    dimx = getdimension(trainingdata)
    xtrain = get_original_x(trainingdata)
    ytrain = get_original_y(trainingdata)

    println("\n\n----GP model information----\n")
    println("   kernel (params in log scale): $(pred_result.model.kernel)")
    println("   Model fitting status: $(Optim.converged(pred_result.res))")
    println("   Optimal parameters: \n      $(pred_result.likelihood_optimum)")
    println("\n\n----GP Prediction Results----\n")
    println("   Number of training data: ", ntrain)
    println("   Number of testing data: ", ntest)
    println("   Mean squared error = $(mean(pred_sq_err))")
    println("   Mean absolute error = $(mean(pred_abs_err))")
    println("   Mean negative log predictive density = $(mean(pred_nlpd))")
    println("   CI accuracy = $(pred_result.CI_accuracy)")
    println("   Time cost: \n       $(pred_time_cost)")

    if verbose
        print("Verbose Mode On... ") 
        @show pred_mean
        @show pred_stdv
        @show pred_abs_err
        @show pred_sq_err
        @show pred_nlpd
        @show pred_time_cost
    end

    z = 1.96 
    if confidence_level != 0.95
        @warn "confidence_level != 0.95, need to change z value"
    end
    
    
    
    if plot_results
        if dimx == 1
            xtest = dropdims(x0, dims=2)
            PyPlot.close("all") #close existing windows
            PyPlot.plot(x0, pred_mean, label = "mean")
            PyPlot.plot(x0, y0_true, label = "true")
            PyPlot.fill_between(xtest, -z.*pred_stdv .+ pred_mean, z.*pred_stdv .+ pred_mean, alpha = 0.3, label = "$(ceil(100*confidence_level))% confidence interval")
            PyPlot.scatter(xtrain, ytrain, s = 10, c = "k", marker = "*")
            PyPlot.legend(fontsize=8)
            if figure_title != nothing 
                PyPlot.title(figure_title, fontsize=10)
            end
            if save_path != nothing
                PyPlot.savefig("$(save_path).pdf")
                println("Figure saved: $(save_path).pdf")
            end
        else
            @info "No plots for high dimensional data"
        end
    end
end
            


    

