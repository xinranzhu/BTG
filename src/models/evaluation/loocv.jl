
"""
"""
function loocv(btg; method = "naive")
    R_i = zeros(n) #remainders
    if method == "naive"
        train = btg.trainingdata
        n = train.n
        for i = 1:n
            trainingdata_i = leaveOut(train, i)
            # build submodel using trainingdata_i
            #btg_optimize!(mybtg, parameter_names, lower_bound, upper_bound;
            #multistart=NUM_MLE_MULTISTART, randseed=1234, initial_guess=nothing, sobol=false)
            # predict at ith point and compute remainder r_i
            test_point = TestingData(ith(train, i))
        end

    elseif method == "fast"
    else
        error("method not defined")
    end
end
