
"""
Define new iterator which extracts dominant node-weight pairs from a nodes-weights iterator
object.  It is guaranteed that the weights chosen to build the sub-iterator add up to >threshold,
which defaults to 0.99. This is motivated by the observation that many times, the majority of
weights are less than 1e-2. The sub-iterator is rescaled so that its weights sum to 1.

    OUTPUTS:
        - frac: fraction of original weights used
        - sum_new_iter: sum of original weights used

"""
function get_iter_dominant_weights(iter_normalized; threshold = 0.99)
    all_weights = []
    for (_, w) in iter_normalized
        push!(all_weights, w)
    end
    n = length(all_weights)
    sorted_weights = sort(all_weights)
    @assert sorted_weights[n] >= sorted_weights[1]
    cutoff = sorted_weights[n]
    cur_sum = cutoff
    cur_length = 1
    for i = 1:n-1
        if cur_sum > threshold
            break
        else
            cur_sum = cur_sum + sorted_weights[n-i]
            cutoff = sorted_weights[n-i]
            cur_length = cur_length + 1
        end
    end
    frac = cur_length/n #fraction of original weights used
    if frac > 0.5
        @warn "Majority of weights was used in constructing sub-iterator: $frac"
    end
    new_iter = ((n, w) for (n, w) in iter_normalized if w >= cutoff)
    #normalize new_iter
    sum_new_iter = 0.0
    for (n, w) in new_iter
        #@show w
        sum_new_iter = sum_new_iter + w
    end
    final_iter = ((n, w / sum_new_iter) for (n,w) in new_iter)
    return final_iter, frac, sum_new_iter
end

function length_of_iterator(it)
    count = 0
    for i in it
        count += 1
    end
    return count
end
