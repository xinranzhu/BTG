"""
A quadrature domain specifies the domain of integration.
"""
abstract type AbstractQuadratureDomain end

#TODO Add types for function inp

function get_acc_dimensions_iterator(ordered_parameters)
    ordered_parameter_dimensions = [getdimension(getparameter(p.second)) for p in ordered_parameters]
    acc_ordered_parameter_dimensions = ones(length(ordered_parameter_dimensions) + 1)
    for i = 2:length(ordered_parameters)
        acc_ordered_parameter_dimensions[i:end] = acc_ordered_parameter_dimensions[i:end] .+ ordered_parameter_dimensions[i-1]
    end
    acc_ordered_parameter_dimensions[end] = acc_ordered_parameter_dimensions[end-1] + ordered_parameter_dimensions[end]
    return acc_ordered_parameter_dimensions
end

function ordered_parameter_to_merged_parameter_index(ordered_parameters, filtered_quad_info)::Array{T where T<:Int64}
    """
    Assume ordered and merged parameters are sorted by QuadType
    """
    n = length(ordered_parameters)
    ordered_to_merged = ones(1, n)
    ordered_quad_types = getquadtype.((x->x.second).(Iterators.flatten(filtered_quad_info)))
    for i = 2:n
        if ordered_quad_types[i] == ordered_quad_types[i-1]
            ordered_to_merged[i] = ordered_to_merged[i-1]
        else
            ordered_to_merged[i] = ordered_to_merged[i-1] + 1
        end
    end
    return ordered_to_merged
end

function merged_parameter_to_parameter_indices(ordered_parameters, ordered_merged_parameter_list, filtered_quad_info)
    map = ordered_parameter_to_merged_parameter_index(ordered_parameters, filtered_quad_info)
    flat_map = [map[i] for i = 1:length(ordered_parameters)]
    m_len = length(ordered_merged_parameter_list)
    ret = []
    for i = 1:m_len
        push!(ret, Pair(findfirst(x -> x == i, flat_map), findlast(x -> x == i,flat_map)))
    end
    return ret
end

function get_local_acc_dimensions_iterator(ordered_parameters, ordered_merged_parameter_list, filtered_quad_info)
    n = length(ordered_parameters)
    m = length(ordered_merged_parameter_list)
    local_inds = merged_parameter_to_parameter_indices(ordered_parameters, ordered_merged_parameter_list, filtered_quad_info)
    t = get_acc_dimensions_iterator(ordered_parameters)
    ret = []
    for i = 1:m
        first = local_inds[i].first
        second = local_inds[i].second
        cur = [t[i] for i = first:second+1]
        cur = cur .- t[first] .+ 1
        push!(ret, (x -> Int64(x)).(cur))
    end
    return ret
end

function get_local_parameter_index(ordered_parameters, ordered_merged_parameter_list, filtered_quad_info)
    iter  = get_local_acc_dimensions_iterator(ordered_parameters, ordered_merged_parameter_list, filtered_quad_info)
    map = ordered_parameter_to_merged_parameter_index(ordered_parameters, filtered_quad_info)
    n = length(ordered_parameters)
    ret = collect(1:length(ordered_parameters))
    for i = 2:n
        if map[i] != map[i-1]
            ret[i:end] = ret[i:end] .- ret[i] .+ 1
        end
    end
    return ret
end

struct QuadratureDomain <: AbstractQuadratureDomain
    """
    quad_specs: implicitly defines integration domain
    get_iterator: obtain a custom iterator for iterating over nodes and weights
    """
    quad_specs::QuadratureSpecs
    ordered_parameters::Array{Pair{String,QuadInfo}}
    filtered_quad_info #TODO: add type
    filtered_ordered_parameters::Array{Array{Parameter,1},1}
    filtered_quad_types::Array{QuadType,1}
    merged_parameter_list #TODO: add type
    function QuadratureDomain(quad_specs::QuadratureSpecs)
        ordered_parameters = sort(collect(quad_specs.quad_info_dict), by = a -> getquadtype(a.second))
        filtered_quad_info = [] #QuadInfo list, grouped by QuadType
        for c in groupby(x -> getquadtype(x.second), ordered_parameters)
           push!(filtered_quad_info, c)
        end
        filtered_ordered_parameters = [getparameter.((x -> x.second).(t)) for t in filtered_quad_info]
        filtered_quad_types =  [getquadtype(x[1].second) for x in filtered_quad_info]
        merged_parameter_list = [] #also induces ordering
        for i = 1:length(filtered_quad_info)
            pm_list = [getparameter(ele.second) for ele in filtered_quad_info[i]]
            mp = merge(pm_list) #merged parameter;
            push!(merged_parameter_list, mp)
        end

        return new(quad_specs, ordered_parameters, filtered_quad_info,filtered_ordered_parameters,
                filtered_quad_types, merged_parameter_list)
    end
end

getparameters(domain::AbstractQuadratureDomain) = domain.quad_specs.parameters

"""
This function takes the number of nodes per dimension
for each merged parameter to be the max of the num_nodes of its constituent
elements (for Gaussian Quadrature). For MC and QMC, the total number of
nodes of the fixed parameter is the product of the num_nodes of its constituent
elements. For Sparse Grid, we take the max of the levels of each constituent
parameter.
"""
function get_iterator(qd::AbstractQuadratureDomain; verbose=false) #iterates over named parameter first
    #get nodes and weights for agglomerated parameters
    merged_parameter_nw_dict = Dict()
    for i = 1:length(qd.merged_parameter_list)
        cur_parameter = qd.merged_parameter_list[i]
        cur_quad_type = qd.filtered_quad_types[i]

        cur_num_nodes = 0;
        cur_num_levels = 0;

        if (cur_quad_type == QuadType(1) || cur_quad_type ==  QuadType(2)) #MC and QMC
            cur_num_nodes = prod(Base.map(x -> (x.second).num_points, qd.filtered_quad_info[i]))
        elseif cur_quad_type==QuadType(0)
                cur_num_nodes = max(Base.map(x -> (x.second).num_points, qd.filtered_quad_info[i])...)
        else #Sparse Grid
                cur_num_levels =  max(Base.map(x -> (x.second).levels, qd.filtered_quad_info[i])...)
        end

        #TODO pass in default num_nodes and levels somewhere
        merged_parameter_nw_dict[i] =
        get_nodes_weights(cur_quad_type, getrange(cur_parameter);
            num_nodes = cur_num_nodes, levels = cur_num_levels, 
            verbose=verbose)
    end

    heights = [size(merged_parameter_nw_dict[i], 1) for i=1:length(qd.merged_parameter_list)]
    map = ordered_parameter_to_merged_parameter_index(qd.ordered_parameters, qd.filtered_quad_info)
    lind = get_local_parameter_index(qd.ordered_parameters, qd.merged_parameter_list, qd.filtered_quad_info)
    iter = get_local_acc_dimensions_iterator(qd.ordered_parameters, qd.merged_parameter_list, qd.filtered_quad_info)

    nodes_iter = (Dict(zip(first.(qd.ordered_parameters), cat([[merged_parameter_nw_dict[map[i]][x[map[i]],iter[map[i]][lind[i]]:(iter[map[i]][lind[i]+1]-1)]]
                for i=1:length(qd.ordered_parameters)]..., dims=2)))
                for x in Iterators.product([collect(1:num) for num in heights]...))
    weights_iter = (prod([merged_parameter_nw_dict[map[i]][x[map[i]], end]
                for i=1:length(qd.merged_parameter_list)])
                for x in Iterators.product([collect(1:num) for num in heights]...))
    nw_iter = ((mapping, weight) for (mapping, weight) in zip(nodes_iter, weights_iter))
    return nw_iter
end

"""
Inputs
    it: the iterator from get_iterator, including nodes and weights from quadrature rule
    likelihood: the likelihood of training data given parameter value
    prior: the prior of each paramter
Outputs
    it_normalized: another iterator with weights, combining likelihood, prior and quadrature weights, then normalized
"""
function normalized_weight_likelihood_prior_iterator(it::Base.Generator,
    likelihood::Function, prior::Function, buffer::BufferDict, lookup = true)
    # max_log_w = 0; #maximum log of absolute value of weights
    # max_log_likelihood = -Inf;
    max_exponent = -Inf
    for (node, weight) in it
        # @info node
        # @info weight, 1e20

        cur_logabsweight = log(abs(weight))
        # max_log_w = abs(logabsweight) > abs(max_log_w) ? logabsweight : max_log_w
        cur_log_likelihood = likelihood(node, buffer; log_scale = true, lookup = lookup)
        # @info cur_log_likelihood, -Inf
        cur_log_prior = prior(node; log_scale = true)
        # @info cur_log_prior, 0
        sum_cur_exponent = cur_logabsweight + cur_log_likelihood + cur_log_prior
        # max_log_likelihood = max(cur_log_likelihood, max_log_likelihood)
        # max_log_prior = max(cur_log_prior, max_log_prior)
        max_exponent = max(max_exponent, sum_cur_exponent)
    end
    it_temp = ((n, prior(n; log_scale = true) + likelihood(n, buffer; log_scale = true)
        + log(abs(w)) - max_exponent, sign(w)) for (n, w) in it)
    # for (n, w, s) in it_temp
    #     @show w
    # end
    sum_exp_log_abs_weights = sum([exp(w)*s for (_, w, s) in it_temp])
    # @info sum_exp_log_abs_weights
    it_normalized = (( n, exp(w)/sum_exp_log_abs_weights*s) for (n, w, s) in it_temp)
    return it_normalized
end

function integrate(f::Function, qd::AbstractQuadratureDomain; iterator = nothing)
    iter = iterator == nothing ? get_iterator(qd) : iterator
    total = 0.0
    for (n, w) in iter  #can parallelize this part?
        cur = f(n)[1]*w[1]
        total += cur[1]
    end
    return total
end

"""
build iterator out of likelihood, prior and quadrature weights

Outputs:
- normalized_weight_likelihood_prior_iterator

TODO: test this function in a notebook/test file
"""
function build_normalized_weight_iterator(
    train::TrainingData,
    likelihood::Function,
    domain::QuadratureDomain,
    transform::AbstractTransform,
    kernel_module::KernelModule,
    buffer::BufferDict;
    lookup = true,
    drop_weights = false,
    drop_threshold = 1e-5,
    num_weights = nothing,
    total_mass = nothing,
    verbose=false
    )
    xtrain = getposition(train)
    ytrain = getlabel(train)
    function likelihood_fixed_train(d, buffer::BufferDict; log_scale=false, lookup = lookup) #TODO typecheck d
        d_copy = Dict()
        for k in keys(d)
            if length(d[k]) == 1
                d_copy[k] = d[k][1] #remove bracket for scalars
            else
                d_copy[k] = d[k]
            end
        end
        #reformat d so that it is compatible with likelihood (becomes nested with keyword arg COMPOSED_TRANSFORM_DICTS)
        #REQUIRES names of d to follow specfic format, i.e. COMPOSED_PARAMETER_i
        return likelihood(reformat(d_copy, transform), train, transform, kernel_module, buffer; log_scale=log_scale, lookup = lookup)
    end
    it = get_iterator(domain; verbose=verbose)
    parameters = domain.quad_specs.parameters
    # check sum of quadrature weights
    # @show "check quadrature weights ", sum([w for (_, w) in it])
    function fixed_prior(d; log_scale=false)
        list_priors = Dict(p.name => p.prior for p in parameters)
        total = 0
        for key in keys(d)
            if list_priors[key] != nothing
                total += evaluate(list_priors[key], d[key]; log_scale=log_scale)
            end
        end
        return total
    end
    it_normalized =
        normalized_weight_likelihood_prior_iterator(it, likelihood_fixed_train, fixed_prior, buffer)
    # check if sum(w) = 1
    sum_w = sum([w for (_, w) in it_normalized])
    # @info "In it_normalized, sum(w)", sum_w
    if verbose
        println("\nNumber of weights before drop-out: $(length_of_iterator(it_normalized))")
    end

    if total_mass != nothing
        cur_ratio = 0
        cur_sum = 0
        cur_abs_sum = 0
        idx = 0
        nw = [item for item in it_normalized]
        nw = sort(nw, by=x->abs(x[2]))[end:-1:1]
        total_abs_sum = sum([abs(item[2]) for item in it_normalized])
        while cur_ratio < total_mass-1e-10
            idx += 1
            cur_abs_sum += abs(nw[idx][2])
            cur_sum += nw[idx][2]
            cur_ratio =  cur_abs_sum/total_abs_sum
        end
        it_normalized = ((n, w/cur_sum) for (n,w) in nw[1:idx])
        if verbose
            println("\nTo select $(total_mass) total mass, selected $(idx) nodes.")
        end
    end

    if drop_weights == true
        d = Dict()
        for (n, w) in it_normalized
            d[n] = w
        end
        weights_drop_iter = ((k, d[k]) for k in keys(d) if abs(d[k]) > drop_threshold)
        cur_sum = sum([w for (_, w) in weights_drop_iter])
        if verbose
            println("\nNumber of weights after drop-out: $(length_of_iterator(weights_drop_iter))")
        end
        it_normalized = ((n, w/cur_sum) for (n, w) in weights_drop_iter)
    end
    
    if num_weights != nothing #specify the number of weights a priori (only for timing purposes)
        arr = []
        for (n, w) in it_normalized
            push!(arr, (n, w))
        end
        x = sort(arr, by = x->abs(x[2]))
        x = reverse(x)
        if num_weights > length(x)
            error("number specified weights is greater than total number of weights")
        end
        x_select = x[1:num_weights]
        x_left = x[num_weights+1:end]
        weights_arr = [item[2] for item in x_select]
        weights_left = [item[2] for item in x_left]
        # println("\n weights_arr = $(weights_arr) ")
        # println("\n weights_left = $(weights_left) ")
        nodes_arr = [item[1] for item in x_select]
        total_weight = sum(weights_arr)
        select_weight_percent = sum(abs.(weights_arr))/(sum(abs.(weights_arr)) + sum(abs.(weights_left)))
        if num_weights > 0
            if verbose
                println("\nTo pick $num_weights nodes, selected weights percentage = $(round(select_weight_percent, digits=4))")
            end
        end
        it_normalized = ((nodes_arr[i], weights_arr[i]/total_weight) for i=1:num_weights)
    end
    return it_normalized

end

"""
Stores values from likelihood iterator so that they are not recomputed
each time the generator is used. Returns fixed iterator.
"""
function fixed_generator(nw_iterator)
    d = Dict()
    for (n, w) in nw_iterator
        d[n] = w
    end
    return ((k, d[k]) for k in keys(d))
end
