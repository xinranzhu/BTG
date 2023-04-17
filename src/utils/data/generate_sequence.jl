function sample_x(N_sample, lb, ub; method = "sobol")
    if method == "sobol"
        s = SobolSeq(lb, ub) # length d
        N = hcat([next!(s) for i = 1:N_sample]...)'  # N by d
        return N
    else
        error("sampling method not defined")
    end
end

function sample_f(OBJECTIVE, OBJ_RANGE; n = 15, method = "sobol")
    MINIMIZER = 1.0
    dimx = 1
    #burn-in
    INIT_X = convert(Array{Float64, 2}, sample_x(n, OBJ_RANGE[:, 1][:], OBJ_RANGE[:, 2][:], method = method))
    INIT_FX = INIT_X
    INIT_Y = dropdims(mapslices(OBJECTIVE, INIT_X, dims=2), dims=2)
    td = TrainingData(INIT_X, INIT_X, INIT_Y)
end
