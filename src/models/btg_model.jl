"""
Args
    - train: TrainingData
    - domain: integraion domain
    - conditional_posterior_pdf: p( y0 | y, w)
    - likelihood: p(w | y)
    - iter_normalized: iterator for prior*quad_weight*likelihood at each node
    - posterior_pdf: posterior probability density function
    - posterior_cdf: posterior cumulative density function
    - buffer: buffer dictionary for all stored buffers (large units) and caches (small units)
    - lookup: bool saying whether or not to make use of buffers, for debugging/timing purposes

Upon instantiation: creates BufferDict for storing precomputed quantities
"""
mutable struct Btg
    trainingdata::TrainingData
    kernel::KernelModule
    transform::AbstractTransform
    domain::QuadratureDomain
    modelname::String # WarpedGP or Btg
    conditional_posterior::Function
    likelihood::Function
    log_likelihood_derivatives::Function
    iter_normalized::Base.Generator
    iter_dominant_weights::Base.Generator
    iter_dominant_frac::Number
    iter_dominant_sum::Number
    posterior_pdf::Function
    posterior_cdf::Union{Function, Counter}
    quantile_bound::Function
    quantile_bound_single_weight::Function
    quantile_bound_dominant_weights::Function
    quantile_bound_composite::Function
    MLE_optimize_status::Union{Nothing, Bool}
    likelihood_optimum::Pair{A, T} where A<:Dict where T<:Float64
    get_mean_and_std
    buffer::BufferDict
    lookup::Bool
    Debug::Any
    function Btg(train::TrainingData, domain::QuadratureDomain, modelname::String,
        # conditional_posterior::Function, likelihood::Function, log_likelihood_derivatives::Function,
        transform::AbstractTransform,
        kernel::KernelModule;
        lookup = true,
        drop_weights = false,
        drop_threshold = 1e-5,
        num_weights = nothing,
        total_mass = nothing,
        verbose = false,
        )
        #Debug = Dict("pdf_calls"=>0, "cdf_calls"=>0)
        # compute iter_normalized, posterior_pdf
        if verbose
            println("\nBTG Dropout setups: drop_weights = $drop_weights, drop_threshold=$(drop_threshold), num_weights = $num_weights, total_mass = $total_mass.")
        end
        buffer = BufferDict()
        # use modelname to call different models
        likelihood = eval(Symbol("$(modelname)Likelihood"))
        log_likelihood_derivatives = eval(Symbol("$(modelname)LogLikelihoodDerivatives"))
        conditional_posterior = eval(Symbol("$(modelname)ConditionalPosterior"))
        iter_non_fixed = #instructions for computing static iterator
            build_normalized_weight_iterator(train, likelihood, domain, transform, kernel,
                buffer, lookup = lookup, drop_weights = drop_weights, drop_threshold = drop_threshold, 
                num_weights = num_weights, total_mass = total_mass, verbose=verbose)
        iter_normalized = fixed_generator(iter_non_fixed) #fix/store values
        if verbose == true
            cc = collect(iter_normalized)
            w = [item[2] for item in cc]
            #sort(w)
            #@info "Top 5 weights are: $(w[1]),  $(w[2]),  $(w[3]),  $(w[4]),  $(w[5]) "
            #@info "Average Weight is: $(mean(w))"
        end
        iter_dominant_weights, iter_dominant_frac, iter_dominant_sum = get_iter_dominant_weights(iter_normalized, threshold = 0.9)
        #TODO: type check of every input
        function get_mean_and_std(x0, Fx0)
            arr = []
            for (node_dict, w) in iter_normalized
                moments = conditional_posterior(node_dict, x0, Fx0, 0.5, train, transform, kernel, buffer)[4]
                push!(arr, (node_dict, w, moments))
            end
            sort(arr, by = x -> x[2]) #sort by weight
            return arr
        end

        integrand_pdf(d, x0, Fx0, y0; lookup = lookup, test_buffer = Buffer()) =
            @timeit to "integrand pdf eval" begin
            conditional_posterior(d, x0, Fx0, y0, train, transform, kernel, buffer;
                lookup = lookup, test_buffer = test_buffer)[1]
            end
        integrand_cdf(d, x0, Fx0, y0; lookup = lookup, test_buffer = Buffer()) =
            @timeit to "integrand cdf eval" begin
            conditional_posterior(d, x0, Fx0, y0, train, transform, kernel, buffer;
                lookup = lookup, test_buffer = test_buffer)[2]
            end
        quantile_ij(d, x0, Fx0, y0, quant; lookup = lookup, test_buffer = Buffer()) =
            conditional_posterior(d, x0, Fx0, y0, train, transform, kernel, buffer;
                lookup = lookup, test_buffer = test_buffer, quant=quant)[3]
        distr_moments(d, x0, Fx0, y0, quant; lookup = lookup, test_buffer = Buffer()) =
            conditional_posterior(d, x0, Fx0, y0, train, transform, kernel, buffer;
                lookup = lookup, test_buffer = test_buffer, quant=quant)[4]

        function posterior_pdf(x0, Fx0, y0; lookup = lookup, test_buffer = Buffer(), iter = iter_normalized)
            #Debug["pdf_calls"] = Debug["pdf_calls"] + 1
            h(d) = integrand_pdf(d, x0, Fx0, y0; lookup = lookup, test_buffer = test_buffer)
            #@timeit to "integrate eval pdf"
            res = integrate(h, domain; iterator = iter) #domain input is superfluous
            return res
        end
        function _posterior_cdf(x0, Fx0, y0; lookup = lookup, test_buffer = Buffer(), iter = iter_normalized)
            #Debug["cdf_calls"] = Debug["cdf_calls"] + 1
            if !lookup
                @warn lookup
            end
            h(d) = integrand_cdf(d, x0, Fx0, y0; lookup = lookup, test_buffer = test_buffer)
            #@timeit to "integrate eval cdf"
            res = integrate(h, domain; iterator = iter)
            return res
        end
        posterior_cdf = Counter(_posterior_cdf)
        function quantile_bound(x0, Fx0, y0, quant; lookup = lookup, test_buffer = Buffer(), iter = iter_normalized)
            # loop over all quadrature nodes, instead of computing integrand, find the min and max quantile value.
            min_q = Inf; max_q = -Inf
            for (d, _) in iter  #can parallelize this part?
                d_copy = Dict()
                for k in keys(d)
                    if length(d[k]) == 1
                        d_copy[k] = d[k][1] #remove bracket for scalars
                    else
                        d_copy[k] = d[k]
                    end
                end
                q_ij = quantile_ij(d_copy, x0, Fx0, y0, quant; lookup = lookup, test_buffer = test_buffer)
                min_q = min(min_q, q_ij)
                max_q = max(max_q, q_ij)
            end
            if min_q == max_q
                min_q = max(0, min_q*0.9)
                max_q = max_q*1.1
            end
            return [max(min_q, 0), max_q]
        end
        # TODO implement second quantile bound
        function quantile_bound_single_weight(x0, Fx0, y0, quant; lookup = lookup, test_buffer = Buffer(), iter = iter_normalized)
            # loop over all quadrature nodes, instead of computing integrand, find the min and max quantile value.
            min_q = -Inf; max_q = Inf
            for (d, w) in iter  #can parallelize this part?
                d_copy = Dict()
                for k in keys(d)
                    if length(d[k]) == 1
                        d_copy[k] = d[k][1] #remove bracket for scalars
                    else
                        d_copy[k] = d[k]
                    end
                end
                bb = quant + 1 - w #upperbound
                aa = quant - (1-w) #lowerbound
                q_bb = bb <= 1 ? quantile_ij(d_copy, x0, Fx0, y0, bb; lookup = lookup, test_buffer = test_buffer) : 1
                q_aa = aa > eps() ? quantile_ij(d_copy, x0, Fx0, y0, aa; lookup = lookup, test_buffer = test_buffer) : eps()
                min_q = max(min_q, q_aa)
                max_q = min(max_q, q_bb)
            end
            if min_q == max_q
                min_q = max(0, min_q*0.9)
                max_q = max_q*1.1
            end
            return [max(min_q, 0), max_q]
        end
        function quantile_bound_dominant_weights(threshold) #wrapper function for setting dominance_weight_threshold
            function inner_quantile_bound_dominant_weights(x0, Fx0, y0, quant; lookup = lookup, test_buffer = Buffer(),
                iter = iter_dominant_weights, dominant_weight_threshold = threshold)
                if dominant_weight_threshold != nothing
                    cur_iter_dominant_weights, cur_iter_dominant_frac, cur_iter_dominant_sum =
                        get_iter_dominant_weights(iter_normalized, threshold = dominant_weight_threshold)
                end
                if iter_dominant_frac > 0.5
                    error("Should not use dominant weight quantile bound, because fraction of dominant weights is $iter_dominant_frac")
                end
                u = 2* ( 1 - iter_dominant_sum )
                lower = max(0, quant - u)
                upper = min(1, quant + u)
                @assert lower < upper
                quantile_bound_fun = x -> quantile_bound(x0, Fx0, y0, x; lookup = lookup, test_buffer = Buffer(), iter = cur_iter_dominant_weights)

                mycdf = y -> posterior_cdf(x0, Fx0, y, iter = iter)

                q_l = quantile(mycdf, [0.0, 5.0], pdf = nothing, pdf_deriv = nothing, p = lower, quantile_bound= nothing,#quantile_bound_fun,
                verbose = false, loose_bound = false, quantile_method = "0order")
                q_u = quantile(mycdf, [0.0, 5.0], pdf = nothing, pdf_deriv = nothing, p = upper, quantile_bound= nothing, #quantile_bound_fun,
                verbose = false, loose_bound = false, quantile_method = "0order")
                #@assert q_l[1] <= q_u[1]
                if q_l[1] > q_u[1]
                    @warn "q_l is $(q_l[1]), while q_u is $(q_u[1])"
                    @warn "lower is $lower while upper is $upper"
                    @warn "u is $u"
                    @warn "iter_dominant_frac is $iter_dominant_frac"
                end
                return [q_l[1], q_u[1]]
            end
            return inner_quantile_bound_dominant_weights
        end
        """
        Apply all quantile bounds and take best one for each point
        """
        function quantile_bound_composite(x0, Fx0, y0, quant; lookup = lookup, test_buffer = Buffer())
            aa = quantile_bound(x0, Fx0, y0, quant; lookup = lookup, test_buffer = test_buffer, iter = iter_normalized)
            bb = quantile_bound_single_weight(x0, Fx0, y0, quant; lookup = lookup, test_buffer = test_buffer, iter = iter_normalized)
            cc = quantile_bound_dominant_weights(x0, Fx0, y0, quant; lookup = lookup, test_buffer = test_buffer, iter = iter_dominant_weights)
            return [max(max(aa[1], bb[1]), cc[1]), min(min(aa[2], bb[2]), cc[2])] #form tightest bound
        end
        # println("\n Model Initialized: $(modelname)")
        return new(train, kernel, transform, domain, modelname, conditional_posterior,
        likelihood, log_likelihood_derivatives, iter_normalized, iter_dominant_weights,
        iter_dominant_frac, iter_dominant_sum, posterior_pdf, posterior_cdf,
        quantile_bound, quantile_bound_single_weight,
        quantile_bound_dominant_weights, quantile_bound_composite, nothing, Dict()=>0.0,
        get_mean_and_std, buffer, lookup, nothing)
    end
end

function examine_weights(btg; verbose=false)
    iter_normalized = btg.iter_normalized
    cc = collect(iter_normalized)
    nw = [item for item in cc]
    nw = sort(nw, by=x->abs(x[2]))[end:-1:1]
    w = [item[2] for item in nw]
    nodes = [item[1] for item in nw]
    print("\n Selected number of Weights: $(length(w))")
    print("\n Avg Weight: $(mean(w))")
    print("\n Max Weight: $(w[1])")
    print("\n Min Weight: $(w[end])")
    if verbose
        if length(w) > 5
            print("\n Top Weights: $(w[1:5]')")
        else
            print("\n Top Weights: $(w')")
        end
        print("\n Corresponding nodes: ")
        for i in 1:min(length(nodes), 5)
            print("\n Node number $(i): $(nodes[i])")
        end
    end

    #@info "Top 5 weights are: $(w[1]),  $(w[2]),  $(w[3]),  $(w[4]),  $(w[5]) "
    #@info "Average Weight is: $(mean(w))"
end

function subset_weights(mybtg, n)
    it_normalized = mybtg.iter_normalized
    arr = []
    for (n, w) in it_normalized
        push!(arr, (n, w))
    end
    x = sort(arr, by = x->abs(x[2]))
    x = reverse(x)
    if n > length(x)
        @info length(x)
        @info n
        error("In subset_weights: number specified weights is greater than total number of weights")
    end
    x = x[1:n]
    weights_arr = [item[2] for item in x]
    nodes_arr = [item[1] for item in x]
    total_weight = sum(weights_arr)
    it_normalized = ((nodes_arr[i], weights_arr[i]/total_weight) for i=1:length(x))
    mybtg.iter_normalized = it_normalized
    return mybtg
end

function reset_counter!(btg)
    btg.posterior_cdf.count = 0
end
