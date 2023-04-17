"""
merge new prediction results into exisiting results
"""
function merge_results!(results::btgResults, results_new::btgResults)
    @assert keys(results.time) == keys(results_new.time) "Results should share same time key"
    @assert keys(results.data) == keys(results_new.data) "Results should share same data key"
    for key in keys(results.time)
        results.time[key] += results_new.time[key]
    end
    for key in keys(results.data)
        append!(results.data[key], results_new.data[key])
    end
end

function merge_posterior!(results::btgPosteriorResults, results_new::btgPosteriorResults)
    @assert keys(results.data) == keys(results_new.data) "Results should share same data key"
    for key in keys(results.data)
        append!(results.data[key], results_new.data[key])
    end
end