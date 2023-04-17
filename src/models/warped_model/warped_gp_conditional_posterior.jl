wgp_posterior_timer = TimerOutput()

#TODO: use Distributions pkg to compute multi-dimenisonal Gaussian cdf and pdf
# for pdf, input g(y) instead of y and mannually add jacobian;
# for cdf, input g(y) instead of y

"""
Computes value of conditional posterior at test point given by (x0, Fx0, y0),
i.e. cdf and pdf values, both of type Float64

INPUTS:
    d - dictionary of hyperparameter name-value mappings
"""
function WarpedGPConditionalPosterior(d, x0, Fx0, y0, trainingdata, transform, kernel_module, buffer_dict;
     lookup = true, test_buffer = Buffer(), quant=nothing, scaling=true)
     #buffer naming convention
     @timeit wgp_posterior_timer "load and define" begin
        covariance_buffer_name = "train_covariance_buffer"
        ntrain = getnumpoints(trainingdata)
        ytrain = reshape(getlabel(trainingdata), ntrain, 1)

        @timeit wgp_posterior_timer "trans input dict" begin
            transform_dict = build_input_dict(transform, reformat(d, transform)) #when there are multiple transforms, how do we pass this information?
        end
        g(y) = evaluate(transform, transform_dict, y)
        jac(y0) = evaluate_derivative_x(transform, transform_dict, y0)
        g_inv(z) = evaluate_inverse(transform, transform_dict, z)
    end
    @timeit wgp_posterior_timer "compute/load covariance matrix" begin
        #%% compute kernel matrix
        @timeit wgp_posterior_timer "cov input dict" begin
            covariance_dict = build_input_dict(kernel_module, d)
        end
        @timeit wgp_posterior_timer "assertions" begin
            @assert "θ" in keys(covariance_dict); @assert "var" in keys(covariance_dict);
            @assert "noise_var" in keys(covariance_dict);
        end
        #if ! typeof(covariance_dict["var"])<:Float64
        #        covariance_dict["var"] = covariance_dict["var"][1]
        #end
        #covariance_dict = Dict("θ" in keys() => θ, "var"=> 1.0)
        @timeit wgp_posterior_timer "lookup cholKθ" begin
            covariance_matrix_cache = lookup_or_compute(buffer_dict, covariance_buffer_name,
             CovarianceMatrixCache(), covariance_dict, trainingdata, kernel_module; lookup = lookup)
            cholKθ = covariance_matrix_cache.chol
        end
    end
    @timeit wgp_posterior_timer "test cache" begin
        #%% compute cross-covariance matrix, etc.
        @timeit wgp_posterior_timer "merge" begin
            test_key_dict = merge(covariance_dict, Dict("x0"=> x0))
            remove_brackets!(test_key_dict) #make arrays of length 1 into floats
        end
        @timeit wgp_posterior_timer "preallocate" begin
            Bθ = zeros(1, 1); Eθ = zeros(1, 1); Dθ = zeros(1, 1); #initialization
        end
        @timeit wgp_posterior_timer "compute Bθ, Eθ, Dθ" begin
            if false
                Bθ = KernelMatrix(kernel_module, x0, getposition(trainingdata); θ = θ)
                Eθ = KernelMatrix(kernel_module, x0; θ = θ, jitter = 0.0)
                Dθ = Eθ - Bθ*(cholKθ\Bθ') .+ covariance_dict["noise_var"]
            else

                test_cache = lookup_or_compute(test_buffer, buffer_dict, TestingCache(),
                    test_key_dict, trainingdata, kernel_module, covariance_buffer_name; lookup = lookup)
            end
        end
        @timeit wgp_posterior_timer "load from test cache" begin
            Bθ = test_cache.Bθ
            Eθ = test_cache.Eθ
            Dθ = test_cache.Dθ
        end
    end
    @timeit wgp_posterior_timer "train cache" begin
        #%% Compute Mean m, which depends on covariance_dict and transform_dict
        # for composed transform, transform_dict = [complete dict]
        @timeit wgp_posterior_timer "upgrade dict" begin
            transform_dict_upgraded = typeof(transform)<:AbstractComposedTransform ? Dict("COMPOSED_TRANSFORM_DICTS"=>transform_dict) : transform_dict
        end
        @timeit wgp_posterior_timer "merge train and cov" begin
            train_key_dict = merge(transform_dict_upgraded, covariance_dict)
        end
        @timeit wgp_posterior_timer "apply nonlin trans" begin
            transform_buffer_name = "transform_buffer"
            #g_of_y = g.(ytrain)
            transform_cache = lookup_or_compute(buffer_dict, "transform_buffer", TransformCache(), transform_dict,
                   transform, ytrain)
            g_of_y = transform_cache.g_of_y
        end
        m = 0
        if false
            cholKθ_inv_g_of_y = cholKθ\g_of_y
            m = (Bθ*cholKθ_inv_g_of_y)[1]
        else
            @timeit wgp_posterior_timer "lookup train cache" begin
                training_cache = lookup_or_compute(buffer_dict, "train_buffer", TrainingCache(),
                    train_key_dict, trainingdata, kernel_module, g_of_y, covariance_buffer_name; lookup = true)
            end
            @timeit wgp_posterior_timer "load Σ_inv_y" begin
                    Σ_inv_y = training_cache.Σθ_inv_y
            end
            @timeit wgp_posterior_timer "dot product" begin
                m = (Bθ*Σ_inv_y)[1]
            end
        end
    end
    @timeit wgp_posterior_timer "posterior dist eval" begin
        posterior_normal = Normal(m, sqrt(Dθ[1]))
        posterior_pdf = Distributions.pdf(posterior_normal, g(y0)) * abs(jac(y0))
        posterior_cdf = Distributions.cdf(posterior_normal, g(y0))
    end
    # assuming all transforms are monotonically increasing, i.e. b>0, lambda > 0
    # then we always truncate from left and the scaling value is always CDF(g(0))!
    if scaling
        # @warn "Scaling is off in warped_gp_conditional_posterior"
        scaling_value =  Distributions.cdf(posterior_normal, g(0.))
        #@info scaling_value
        posterior_pdf /= (1-scaling_value)
        posterior_cdf = (posterior_cdf - scaling_value) / (1-scaling_value)
        if quant != nothing
            quant = quant - scaling_value * quant + scaling_value
         end
    end
    # PDF(y) = sum_ij w_ij F_ij(g_j(y0))
    # z_ij easy to get, where F_ij(q_ij) = q
    # from z_ij get q_ij, where z_ij = g_ij(q_ij)
    # Then y* in [min q_ij, max q_ij]
    # @show begin
    #     "Inside conditional posterior"
    #     d
    #     m, sqrt(Dθ[1])
    #     z_ij = Distributions.quantile(posterior_normal, quant)
    # # end
    # @show quant
    # @show z_ij = Distributions.quantile(posterior_normal, quant)
    # @show Distributions.cdf(posterior_normal, z_ij)
    # @show  g_inv(z_ij)
    @timeit wgp_posterior_timer "quantile single eval" begin
        q_ij = quant == nothing ? nothing : (z_ij = Distributions.quantile(posterior_normal, quant); g_inv(z_ij))
    end
    return posterior_pdf, posterior_cdf, q_ij, (m, sqrt(Dθ[1]))
end
