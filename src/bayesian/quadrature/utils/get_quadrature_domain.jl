#quadrature domain
"""
Domain includes parameters θ, var, and noise_var and random settings for ranges
"""
function getQuadratureDomain(; θ_dim = 1)
        range_θ = repeat(reshape([0, 100.0], 1, 2), θ_dim, 1)
        θ = Parameter("θ", θ_dim, range_θ)
        var = Parameter("var", 1, reshape([8.598, 38.599], 1, 2))
        noise_var = Parameter("noise_var", 1, reshape([2.256053,7.056054], 1, 2))

        params =[θ, var, noise_var] # list of Parameter
        m = Dict("θ"=>QuadInfo(θ; quad_type = QuadType(3), levels=3, num_points=1),
                "var"=>QuadInfo(var; quad_type = QuadType(3), levels=3, num_points=1),
                "noise_var"=>QuadInfo(noise_var; quad_type = QuadType(3), levels=3, num_points=1))
        qs = QuadratureSpecs(params, m)
        DEFAULT_QD = QuadratureDomain(qs);
        return DEFAULT_QD
end
