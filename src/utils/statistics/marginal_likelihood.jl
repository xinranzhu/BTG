function marginal_likelihood(x, name, cur_btg; ind = 1)
    cur_d = deepcopy(cur_btg.likelihood_optimum.first)
    if typeof(cur_d[name])<:Array
        cur_d[name][ind] = x
    else
        cur_d[name] = x
    end
    return cur_btg.likelihood(cur_d, cur_btg.trainingdata, cur_btg.transform, cur_btg.kernel,
    BufferDict(); log_scale=true, lookup = true)
end
