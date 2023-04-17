"""
Outputs:
    - p(y0|w, y), where w is hyperparameters, such as transform hyperparameters
"""
function WarpedGPLikelihood(d, trainingdata::TrainingData, transform, kernel_module::RBFKernelModule,
    buffer::BufferDict; log_scale=false, lookup = true)
    xtrain = getposition(trainingdata);
    ytrain = getlabel(trainingdata);
    @timeit to "likelihood eval" begin
        transform_dict = build_input_dict(transform, d)
        g(y) = evaluate(transform, transform_dict, y)
        #evaluate_derivative_hyperparameters(transform, transform_dict, y)
        #evalute_derivative_x_hyperparameters(transform, transform_dict, y)
        jac(y) = evaluate_derivative_x(transform, transform_dict, y)
        #%% Buffer kernel matrix computation
        #key_dict = Dict("θ" => θ, "var"=> 1.0)
        covariance_dict = build_input_dict(kernel_module, d)
        @assert "θ" in keys(covariance_dict); @assert "var" in keys(covariance_dict); @assert "noise_var" in keys(covariance_dict)
        covariance_matrix_cache = lookup_or_compute(buffer, "train_covariance_buffer", CovarianceMatrixCache(),
            covariance_dict, trainingdata, kernel_module, lookup = lookup) #derivatives = true)
        cholKθ = covariance_matrix_cache.chol

        ntrain = length(ytrain)

        jacobian = sum(log.(abs.(jac.(ytrain))))
        # prior_normal = MvNormal(zeros(ntrain), Matrix(cholKθ)) #TODO check that this doesn't slow things down
        # prior_log_pdf = Distributions.logpdf(prior_normal, g.(ytrain)) + jacobian
        #prior_cdf = Distributions.logcdf(prior_normal, g.(ytrain))
        constant = - ntrain * log(2*pi)/2 - 0.5 * logdet(cholKθ)
        bilinear = - 0.5 * g.(ytrain)' * (cholKθ \ g.(ytrain))
        out = jacobian + constant + bilinear

        return log_scale == true ? out : exp(out)
    end
end

function WarpedGPLogLikelihoodDerivatives(d, trainingdata::TrainingData, transform, kernel_module::RBFKernelModule,
    buffer::BufferDict; lookup = true)::Dict{String, Any}
    xtrain = getposition(trainingdata);
    ytrain = getlabel(trainingdata);
    log_scale=true #always set to true
    @timeit to "deriv loglik eval" begin
        transform_dict =  build_input_dict(transform, d) #when there are multiple transforms, how do we pass this information?
        g(y) = evaluate(transform, transform_dict, y)
        jac(y) = evaluate_derivative_x(transform, transform_dict, y)
        covariance_dict = build_input_dict(kernel_module, d)
        @assert "θ" in keys(covariance_dict); @assert "var" in keys(covariance_dict); @assert "noise_var" in keys(covariance_dict)
        covariance_matrix_cache = lookup_or_compute(buffer, "train_covariance_buffer", CovarianceMatrixCache(),
            covariance_dict, trainingdata, kernel_module, lookup = lookup, derivative = true)

        kernel_d_θ = covariance_matrix_cache.kernel_derivative_θ # array of matrices for multi-dimensional theta
        cholKθ = covariance_matrix_cache.chol
        ntrain = length(ytrain)
        jacobian = sum(log.(abs.(jac.(ytrain))))
        g_of_ytrain = g.(ytrain)
        dg_of_ytrain = jac.(ytrain)
        K_inv_g_of_y = cholKθ \ g_of_ytrain
        dict_derivatives = Dict()
        # Derivatives with respect to various kernel parameters
        if "θ" in keys(d)
            if length(d["θ"]) == 1
                log_constant_deriv = - 0.5 * tr(cholKθ \ kernel_d_θ)
                log_bilinear_deriv = 0.5 * K_inv_g_of_y' * ( kernel_d_θ * K_inv_g_of_y )
                log_jacobian_deriv = 0
                dict_derivatives["θ"] = log_constant_deriv + log_bilinear_deriv + log_jacobian_deriv
            else
                dimθ = length(d["θ"])
                deriv_θ = Any[] # array of same shape as theta
                for k in 1:dimθ
                    kernel_d_θk = kernel_d_θ[k]
                    log_constant_deriv = - 0.5 * tr(cholKθ \ kernel_d_θk)
                    log_bilinear_deriv = 0.5 * K_inv_g_of_y' * ( kernel_d_θk * K_inv_g_of_y )
                    log_jacobian_deriv = 0
                    deriv_θk = log_constant_deriv + log_bilinear_deriv + log_jacobian_deriv
                    push!(deriv_θ, deriv_θk)
                end
                dict_derivatives["θ"] = convert(Array{Float64,1}, deriv_θ)
            end
        end
        if "var" in keys(d)
            I = LinearAlgebra.I(ntrain)
            K_1 = (Matrix(cholKθ) - d["noise_var"][1]*I) / d["var"][1]
            log_constant_deriv = - 0.5 * tr( I/(d["var"][1]) - (cholKθ\((d["noise_var"][1])*I)) / (d["var"][1]) ) #TODO make more efficient
            log_bilinear_deriv = 0.5 * K_inv_g_of_y' * (K_1) * K_inv_g_of_y #TODO make more efficient
            log_jacobian_deriv = 0
            dict_derivatives["var"] = log_constant_deriv + log_bilinear_deriv + log_jacobian_deriv
        end
        if "noise_var" in keys(d)
            I = LinearAlgebra.I(ntrain)
            log_constant_deriv = - 0.5 * tr(cholKθ \ I) #TODO make this more efficient
            log_bilinear_deriv = 0.5 * norm(K_inv_g_of_y)^2
            log_jacobian_deriv = 0
            dict_derivatives["noise_var"] = log_constant_deriv + log_bilinear_deriv + log_jacobian_deriv
        end
        #user defined transform parameters
        if typeof(transform) <: AbstractComposedTransform
            if "COMPOSED_TRANSFORM_DICTS" in keys(d) # see if derivatives w.r.t. parameters are wanted
                my_transform_parameters = d["COMPOSED_TRANSFORM_DICTS"] #array of dicts
                transform_derivatives_func(y) = evaluate_derivative_hyperparams(transform, my_transform_parameters, y)
                transform_second_derivative_func(y) = evaluate_derivative_x_hyperparams(transform, my_transform_parameters, y)
                transform_derivatives = transform_derivatives_func.(ytrain)
                transform_second_derivatives = transform_second_derivative_func.(ytrain)
                array_derivatives = []
                for i = 1:length(my_transform_parameters)
                    cur_derivatives_d = Dict()
                    cur_dict = my_transform_parameters[i]
                    cur_transform_derivatives = [transform_derivatives[k][i] for k=1:ntrain]
                    cur_transform_second_derivatives = [transform_second_derivatives[k][i] for k=1:ntrain]
                    for key in keys(cur_dict)
                        # derivative of g(y) w.r.t transform hyperparameter named key
                        dvec = reshape([cur_transform_derivatives[i][key] for i=1:ntrain], ntrain, 1)
                        d2vec = reshape([cur_transform_second_derivatives[i][key] for i=1:ntrain], ntrain, 1)
                        log_constant_deriv = 0.0
                        log_bilinear_deriv = - dvec' * K_inv_g_of_y
                        log_jacobian_deriv = sum( d2vec ./ dg_of_ytrain)
                        cur_derivatives_d[key] = log_constant_deriv[1] + log_bilinear_deriv[1] + log_jacobian_deriv[1]
                    end
                    push!(array_derivatives, cur_derivatives_d)
                end
                dict_derivatives["COMPOSED_TRANSFORM_DICTS"] = array_derivatives
            end
        else #regular non-composed transformtion
            my_transform_parameters = Dict(intersect(transform_dict, d))
            function my_transform_derivatives_func(y)
                evaluate_derivative_hyperparams(transform, my_transform_parameters, y)
            end
            transform_derivatives = my_transform_derivatives_func.(ytrain)
            # mixed second derivatives w.r.t y and hyperparameter variable
            function my_transform_second_derivative_func(y)
                evaluate_derivative_x_hyperparams(transform, my_transform_parameters, y)
            end
            transform_second_derivatives = my_transform_second_derivative_func.(ytrain)
            for key in keys(my_transform_parameters)
                # derivative of g(y) w.r.t transform hyperparameter named key
                dvec = reshape([transform_derivatives[i][key] for i=1:ntrain], ntrain, 1)
                d2vec = reshape([transform_second_derivatives[i][key] for i=1:ntrain], ntrain, 1)
                log_constant_deriv = 0.0
                log_bilinear_deriv = - dvec' * K_inv_g_of_y
                log_jacobian_deriv = sum( d2vec ./ dg_of_ytrain)
                dict_derivatives[key] = log_constant_deriv[1] + log_bilinear_deriv[1] + log_jacobian_deriv[1]
            end
        end
        return dict_derivatives
    end
end
