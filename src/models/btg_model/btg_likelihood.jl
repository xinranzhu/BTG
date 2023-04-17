"""
|XΣX|^-1/2 * |Σ|^-1/2 * q^-(n-p)/2 * J^(1-p/n)
"""
function BtgLikelihood(d, trainingdata::TrainingData, transform, kernel_module::RBFKernelModule,
    buffer::BufferDict; log_scale=false, lookup = true)

    xtrain = getposition(trainingdata);
    ytrain = getlabel(trainingdata);
    @timeit to "btg likelihood eval" begin
        transform_dict = build_input_dict(transform, d)
        g(y) = evaluate(transform, transform_dict, y)
        jac(y) = evaluate_derivative_x(transform, transform_dict, y)

        #Access covariance cache
        covariance_dict = build_input_dict(kernel_module, d)
        @assert "θ" in keys(covariance_dict); @assert "var" in keys(covariance_dict); @assert "noise_var" in keys(covariance_dict)
        covariance_matrix_cache = lookup_or_compute(buffer, "btg_train_covariance_buffer", BTGCovarianceCache(),
            covariance_dict, trainingdata, kernel_module, lookup = lookup) #derivatives = true)
        logdetΣθ = covariance_matrix_cache.logdetΣθ
        logdetXΣX = covariance_matrix_cache.logdetXΣX
        
        #Access training cache
        transform_dict_upgraded = typeof(transform)<:AbstractComposedTransform ? Dict("COMPOSED_TRANSFORM_DICTS"=>transform_dict) : transform_dict
        train_key_dict = merge(transform_dict_upgraded, covariance_dict)
        g_of_y = g.(getlabel(trainingdata))
        train_cache = lookup_or_compute(buffer, "btg training caches", BTGTrainingCache(),
            train_key_dict, trainingdata, kernel_module, g_of_y, "btg_train_covariance_buffer"; lookup = true)
        qtilde = train_cache.qtilde

        #Compute log likelihood
        log_jacobian = sum(log.(abs.(jac.(ytrain))))
        n = getnumpoints(trainingdata)
        p = getcovariatedimension(trainingdata)
        out = - 0.5 * logdetΣθ - 0.5 * logdetXΣX - (n-p)/2 * log(qtilde) + (1-p/n) * log_jacobian

        return log_scale == true ? out : exp(out)
    end
end
