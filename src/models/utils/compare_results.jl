function compare_results(pred_result::btgPrediction, gp_results::btgPrediction;
                        plot_results=false, figure_title=nothing, save_path=nothing, modelname="warped_gp")

    pred_median = pred_result.median
    pred_CI = pred_result.credible_interval
    pred_abs_err = pred_result.absolute_error
    pred_sq_err = pred_result.squared_error
    pred_nlpd = pred_result.negative_log_pred_density
    pred_time_cost = pred_result.time_cost;

    # pred_mean_gp = gp_results.mean
    # pred_stdv_gp = gp_results.stdv
    # pred_abs_err_gp = gp_results.absolute_error
    # pred_sq_err_gp = gp_results.squared_error
    # pred_nlpd_gp = gp_results.negative_log_pred_density
    # pred_time_cost_gp = gp_results.time_cost
    # confidence_level_gp = gp_results.confidence_level

    pred_median_gp = gp_results.median
    pred_CI_gp = gp_results.credible_interval
    pred_abs_err_gp = gp_results.absolute_error
    pred_sq_err_gp = gp_results.squared_error
    pred_nlpd_gp = gp_results.negative_log_pred_density
    pred_time_cost_gp = gp_results.time_cost
    


    testingdata = pred_result.testingdata
    # testingdata_gp = gp_results.testingdata
    # @assert testingdata_gp == testingdata
    x0 = getposition(testingdata)
    y0_true = getlabel(testingdata)
    xtest = dropdims(x0, dims=2)
    ntest = getnumpoints(testingdata)


    trainingdata = pred_result.model.trainingdata
    # trainingdata_gp = gp_results.trainingdata
    # @assert trainingdata == trainingdata_gp
    xtrain = get_original_x(trainingdata)
    ytrain = get_original_y(trainingdata)
    ntrain = getnumpoints(trainingdata)



    if plot_results == true
        # sort 
        idx = sortperm(reshape(x0, 400))
        x0 = x0[idx]
        xtest = xtest[idx]
        y0_true = y0_true[idx]
        pred_median_gp = pred_median_gp[idx]
        pred_median = pred_median[idx]
        pred_CI = pred_CI[idx]
        pred_CI_gp = pred_CI_gp[idx]

        @show sort(xtrain[:,1])
        PyPlot.close("all") #close existing windows
        PyPlot.plot(x0, pred_median, "b-", label = "$modelname median")
        PyPlot.plot(x0, pred_median_gp, "r-", label = "WGP mean")
        PyPlot.plot(x0, y0_true,  label = "true")
        # PyPlot.fill_between(xtest, [x for (x, _) in pred_CI], [y for (_, y) in pred_CI], alpha = 0.3, label = "95% confidence interval")
        PyPlot.plot(xtest, [x for (x,_) in pred_CI], "b.-", label = "95% CI BTG")
        PyPlot.plot(xtest, [y for (_,y) in pred_CI], "b.-")
        PyPlot.plot(xtest, [x for (x,_) in pred_CI_gp], "r:", label = "95% CI WGP")
        PyPlot.plot(xtest, [y for (_,y) in pred_CI_gp], "r:")

        PyPlot.scatter(xtrain, ytrain, s = 20, c = "k", marker = "*")
        PyPlot.legend(fontsize=8)
        if figure_title != nothing
            PyPlot.title(figure_title, fontsize=10)
        end
        if save_path != nothing
            println("$(save_path).pdf")
            PyPlot.savefig("$(save_path).pdf")
            println("Figure saved: $(save_path).pdf")
        end
    end

end
