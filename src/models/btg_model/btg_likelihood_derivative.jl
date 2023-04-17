include("derivatives/btg_likelihood_derivative_helpers.jl")

"""
Derivative of log likelihood, where likelihood is given by

p(z|Θ, Λ) = |XΣX|^-1/2 * |Σ|^-1/2 * q^-(n-p)/2 * J^(1-p/n)

log likelihood is

-0.5 * logdet(XΣX) - 0.5 * logdet(Σ) - (n-p)/2 * log(qtilde) + (1-p/n) * log(J)

where
 1) qtilde = g_of_y'*Σθ_inv_y  - 2*g_of_y'*Σθ_inv_X*βhat + βhat'*Fx'*Σθ_inv_X*βhat
 2) βhat = choleskyXΣX backslash (Fx'*Σθ_inv_y)

Take derivatives w.r.t
a) covariance parameters: θ, var, noise_var
b) transform parameters: ...

"""
function BtgLogLikelihoodDerivatives(d, trainingdata::TrainingData, transform, kernel_module::RBFKernelModule,
    buffer::BufferDict; lookup = true)::Dict{String, Any}
    xtrain = getposition(trainingdata);
    ytrain = getlabel(trainingdata);
    Fx = getcovariate(trainingdata)
    n = getnumpoints(trainingdata)
    p = getcovariatedimension(trainingdata)
    log_scale=true #always set to true
    #Preallocate space to store derivatives
    dict_derivatives = Dict()

    transform_dict =  build_input_dict(transform, d) #when there are multiple transforms, how do we pass this information?
    g(y) = evaluate(transform, transform_dict, y)
    jac(y) = evaluate_derivative_x(transform, transform_dict, y)
    covariance_dict = build_input_dict(kernel_module, d)
    @assert "θ" in keys(covariance_dict); @assert "var" in keys(covariance_dict); @assert "noise_var" in keys(covariance_dict)
    covariance_matrix_cache = lookup_or_compute(buffer, "btg_train_covariance_buffer", BTGCovarianceCache(),
        covariance_dict, trainingdata, kernel_module, lookup = lookup, derivative = true)
    Σθ_prime = covariance_matrix_cache.dΣθ # array of matrices for multi-dimensional theta
    choleskyΣθ = covariance_matrix_cache.choleskyΣθ
    choleskyXΣX = covariance_matrix_cache.choleskyXΣX
    Σθ_inv_X = covariance_matrix_cache.Σθ_inv_X
    jacobian = sum(log.(abs.(jac.(ytrain))))
    g_of_ytrain = g.(ytrain)
    dg_of_ytrain = jac.(ytrain)

    #Access training cache
    transform_dict_upgraded = typeof(transform)<:AbstractComposedTransform ? Dict("COMPOSED_TRANSFORM_DICTS"=>transform_dict) : transform_dict
    train_key_dict = merge(transform_dict_upgraded, covariance_dict)
    g_of_y = g.(getlabel(trainingdata))
    train_cache = lookup_or_compute(buffer, "btg training caches", BTGTrainingCache(),
        train_key_dict, trainingdata, kernel_module, g_of_y, "btg_train_covariance_buffer"; lookup = true)
    qtilde = train_cache.qtilde
    βhat = train_cache.βhat
    Σθ_inv_y = train_cache.Σθ_inv_y

    #preallocate space for derivatives
    dict_derivatives = Dict()

    if "θ" in keys(d)
        if length(d["θ"]) == 1
            (log_qtilde_prime_theta, logdetΣθ_prime_θ, logdetXΣX_prime_θ) = btg_derivatives_per_θ(choleskyΣθ, choleskyXΣX, Σθ_prime, Σθ_inv_X, g_of_y, βhat, qtilde, Fx, Σθ_inv_y)
            AA = -0.5 * logdetΣθ_prime_θ
            BB = -0.5* logdetXΣX_prime_θ
            CC = - (n-p)/2 * log_qtilde_prime_theta
            dict_derivatives["θ"] = (AA .+ BB .+ CC)[1]
        else
            dimθ = length(d["θ"])
            deriv_θ = Any[] # array of same shape as theta
            for k in 1:dimθ
                Σθ_prime_k = Σθ_prime[k]
                (log_qtilde_prime_theta, logdetΣθ_prime_θ, logdetXΣX_prime_θ) = btg_derivatives_per_θ(choleskyΣθ, choleskyXΣX, Σθ_prime_k, Σθ_inv_X, g_of_y, βhat, qtilde, Fx, Σθ_inv_y)
                AA = -0.5 * logdetΣθ_prime_θ
                BB = -0.5* logdetXΣX_prime_θ
                CC = - (n-p)/2 * log_qtilde_prime_theta
                push!(deriv_θ, (AA .+ BB .+ CC)[1])
            end
            dict_derivatives["θ"] = convert(Array{Float64,1}, deriv_θ)
        end
    end
    if "var" in keys(d)
        I = LinearAlgebra.I(n)
        K_1 = (Matrix(choleskyΣθ) - covariance_dict["noise_var"][1]*I) / covariance_dict["var"][1]
        logdetΣθ_prime_var = - 0.5 * tr( I/(covariance_dict["var"][1]) - (choleskyΣθ\((covariance_dict["noise_var"][1])*I)) / (covariance_dict["var"][1]) ) #TODO make more efficient
        logdetXΣX_prime_var = 0.5 * tr(choleskyXΣX \ (Σθ_inv_X' * K_1 * Σθ_inv_X))
        # 1) qtilde = g_of_y'*Σθ_inv_y  - 2*g_of_y'*Σθ_inv_X*βhat + βhat'*Fx'*Σθ_inv_X*βhat
        # 2) βhat = choleskyXΣX backslash (Fx'*Σθ_inv_y)
        σ = covariance_dict["var"]
        σ_noise = covariance_dict["noise_var"]
        #compute βhat_prime_var
        Σθ_inv_y_prime_var = - 1/σ * (Σθ_inv_y - σ_noise * (choleskyΣθ \ Σθ_inv_y))
        Σθ_inv_X_prime_var = - 1/σ * (Σθ_inv_X - σ_noise * (choleskyΣθ \ Σθ_inv_X))
        AA = - (choleskyXΣX \ ( Fx' * (Σθ_inv_X_prime_var * (choleskyXΣX \ (Fx' * Σθ_inv_y)))))
        BB = choleskyXΣX \ ( Fx' * Σθ_inv_y_prime_var)
        βhat_prime_var = AA + BB
        #compute qtilde_prime_var
        CC =  g_of_y' * Σθ_inv_y_prime_var
        DD = - 2*g_of_y'*Σθ_inv_X_prime_var*βhat - 2*g_of_y'*Σθ_inv_X*βhat_prime_var
        EE = 2*βhat_prime_var' * Fx' * Σθ_inv_X * βhat + βhat'*Fx'*Σθ_inv_X_prime_var*βhat
        qtilde_prime_var = (CC + DD + EE)
        log_qtilde_prime_var = - (n-p)/2 * qtilde_prime_var / qtilde
        dict_derivatives["var"] = logdetΣθ_prime_var + logdetXΣX_prime_var + log_qtilde_prime_var
    end
    if "noise_var" in keys(d)
        I = LinearAlgebra.I(n)
        logdetΣθ_prime_noise_var = - 0.5 * tr(choleskyΣθ \ I) #TODO make this more efficient
        logdetXΣX_prime_noise_var = 0.5 * tr(choleskyXΣX \ (Σθ_inv_X'*Σθ_inv_X))

        Σθ_inv_y_prime_noise_var = - (choleskyΣθ \ Σθ_inv_y)
        Σθ_inv_X_prime_noise_var = - (choleskyΣθ \ Σθ_inv_X) #TODO store this, and share with var derivative.

        AA = - (choleskyXΣX \ ( Fx' * (Σθ_inv_X_prime_noise_var * (choleskyXΣX \ (Fx' * Σθ_inv_y)))))
        BB = choleskyXΣX \ ( Fx' * Σθ_inv_y_prime_noise_var)
        βhat_prime_noise_var = AA + BB
        #compute qtilde_prime_var
        CC =  g_of_y' * Σθ_inv_y_prime_noise_var
        DD = - 2*g_of_y'*Σθ_inv_X_prime_noise_var*βhat - 2*g_of_y'*Σθ_inv_X*βhat_prime_noise_var
        EE = 2*βhat_prime_noise_var' * Fx' * Σθ_inv_X * βhat + βhat'*Fx'*Σθ_inv_X_prime_noise_var*βhat
        qtilde_prime_noise_var = (CC + DD + EE)
        log_qtilde_prime_noise_var = - (n-p)/2 * qtilde_prime_noise_var / qtilde

        dict_derivatives["noise_var"] = logdetΣθ_prime_noise_var + logdetXΣX_prime_noise_var + log_qtilde_prime_noise_var
    end
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
                cur_transform_derivatives = [transform_derivatives[k][i] for k=1:n]
                cur_transform_second_derivatives = [transform_second_derivatives[k][i] for k=1:n]
                for key in keys(cur_dict)
                    # derivative of g(y) w.r.t transform hyperparameter named key
                    dvec = reshape([cur_transform_derivatives[i][key] for i=1:n], n, 1)
                    d2vec = reshape([cur_transform_second_derivatives[i][key] for i=1:n], n, 1)
                    log_jacobian_deriv = sum( d2vec ./ dg_of_ytrain)
                    Σθ_inv_d_g_of_y = choleskyΣθ \ dvec
                    dqtilde = qtilde_prime_transform(choleskyΣθ, choleskyXΣX, dvec, Σθ_inv_y, Σθ_inv_X,
                        g_of_y, Fx, Σθ_inv_d_g_of_y, βhat)

                    cur_derivatives_d[key] = (1-p/n) * log_jacobian_deriv[1] - (n-p)/2 * dqtilde[1]/qtilde[1]
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
        # -0.5 * logdet(XΣX) - 0.5 * logdet(Σ) - (n-p)/2 * log(qtilde) + (1-p/n) * log(J)
        for key in keys(my_transform_parameters)
            # derivative of g(y) w.r.t transform hyperparameter named key
            dvec = reshape([transform_derivatives[i][key] for i=1:n], n, 1)
            d2vec = reshape([transform_second_derivatives[i][key] for i=1:n], n, 1)
            log_jacobian_deriv = sum( d2vec ./ dg_of_ytrain)
            #qtilde derivatives
            Σθ_inv_d_g_of_y = choleskyΣθ \ dvec

            dqtilde = qtilde_prime_transform(choleskyΣθ, choleskyXΣX, dvec, Σθ_inv_y, Σθ_inv_X,
                g_of_y, Fx, Σθ_inv_d_g_of_y, βhat)

            dict_derivatives[key] = (1-p/n) * log_jacobian_deriv[1] - (n-p)/2 * dqtilde[1]/qtilde[1]
        end
    end
    return dict_derivatives
end

"""
Derivative of qtilde w.r.t transform parameters.
INPUTS:
 - dvec: derivative of g(y) w.r.t single transform parameter

 1) qtilde = g_of_y'*Σθ_inv_y  - 2*g_of_y'*Σθ_inv_X*βhat + βhat'*Fx'*Σθ_inv_X*βhat
 2) βhat = choleskyXΣX backslash (Fx'*Σθ_inv_y)
"""
function qtilde_prime_transform(choleskyΣθ, choleskyXΣX, dvec, Σθ_inv_y, Σθ_inv_X,
    g_of_y, Fx, Σθ_inv_d_g_of_y, βhat)
    dβhat = choleskyXΣX\(Fx' * Σθ_inv_d_g_of_y)
    AA = 2 * dvec' * Σθ_inv_y
    BB = -2 * dvec' * Σθ_inv_X*βhat - 2 * g_of_y' * Σθ_inv_X * dβhat
    CC = 2 * dβhat'*Fx'*Σθ_inv_X*βhat
    dqtilde = AA + BB + CC
end
