btg_posterior_timer = TimerOutput()

function BtgConditionalPosterior(d, x0, Fx0, y0, trainingdata, transform, kernel_module, buffer_dict;
     lookup = true, test_buffer = Buffer(), quant=nothing, scaling=true)
     @timeit btg_posterior_timer "define and load" begin
         covariance_buffer_name = "btg_train_covariance_buffer"
         train_buffer_name = "btg_train_buffer"
         ntrain = getnumpoints(trainingdata)
         ytrain = reshape(getlabel(trainingdata), ntrain, 1)
         transform_dict = build_input_dict(transform, reformat(d, transform)) #when there are multiple transforms, how do we pass this information?
         g(y) = evaluate(transform, transform_dict, y)
         jac(y0) = evaluate_derivative_x(transform, transform_dict, y0)
         g_inv(z) = evaluate_inverse(transform, transform_dict, z)
     end
     #%% Access Covariance Cache
     @timeit btg_posterior_timer "access covar cache" begin
         covariance_dict = build_input_dict(kernel_module, d)
         @assert "θ" in keys(covariance_dict); @assert "var" in keys(covariance_dict); @assert "noise_var" in keys(covariance_dict);
         covariance_matrix_cache = lookup_or_compute(buffer_dict, covariance_buffer_name,
          BTGCovarianceCache(), covariance_dict, trainingdata, kernel_module; lookup = lookup)
         choleskyΣθ = covariance_matrix_cache.choleskyΣθ
         choleskyXΣX = covariance_matrix_cache.choleskyXΣX
         Σθ_inv_X = covariance_matrix_cache.Σθ_inv_X
     end
     #%% Access Testing Cache
     @timeit btg_posterior_timer "access test cache" begin
         test_key_dict = merge(covariance_dict, Dict("x0"=> x0, "Fx0"=>Fx0))
         test_cache = lookup_or_compute(test_buffer, buffer_dict, BTGTestingCache(),
             test_key_dict, trainingdata, kernel_module, covariance_buffer_name; lookup = lookup)
         Bθ = test_cache.Bθ
         Hθ = test_cache.Hθ
         Cθ = test_cache.Cθ
         transform_dict_upgraded = typeof(transform)<:AbstractComposedTransform ? Dict("COMPOSED_TRANSFORM_DICTS"=>transform_dict) : transform_dict
         train_key_dict = merge(transform_dict_upgraded, covariance_dict)
         g_of_y = g.(ytrain)
     end
     #Access Training Cache
     @timeit btg_posterior_timer "access train cache" begin
         training_cache = lookup_or_compute(buffer_dict, "btg train buffer", BTGTrainingCache(),
             train_key_dict, trainingdata, kernel_module, g_of_y, covariance_buffer_name; lookup = true)
         βhat = training_cache.βhat
         qtilde = training_cache.qtilde
         Σθ_inv_y = training_cache.Σθ_inv_y
     end
     #Compute T-distribution
     @timeit btg_posterior_timer "compute tdist" begin
         n = getnumpoints(trainingdata)
         p = getcovariatedimension(trainingdata)
         m = (Bθ*Σθ_inv_y + Hθ*βhat)[1]
         qC = qtilde[1]*Cθ[1]
         sigma = sqrt(qC/(n-p))
         tdist = LocationScale(m, sigma, TDist(n-p))
         g_of_y0 = (g.(y0))[1]
         jac_of_y0 = (jac.(y0))[1]
         posterior_pdf = Distributions.pdf(tdist, g_of_y0)*jac_of_y0
         posterior_cdf = Distributions.cdf(tdist, g_of_y0)
     end
     if scaling #scaling
        #  @warn "Scaling is on in btg_conditional_posterior"
         scaling_value =  Distributions.cdf(tdist, g(0.))
         posterior_pdf /= (1-scaling_value)
         posterior_cdf = (posterior_cdf - scaling_value) / (1-scaling_value)
         if quant != nothing
            quant = quant - scaling_value * quant + scaling_value
         end
     end
     q_ij = quant == nothing ? nothing : (z_ij = Distributions.quantile(tdist, quant); g_inv(z_ij))
     return posterior_pdf, posterior_cdf, q_ij, (m, sigma, TDist(n-p))
end
