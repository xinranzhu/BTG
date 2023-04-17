"""
Compute derivative of beta hat with respect to theta
"""
function compute_betahat_prime_theta(choleskyΣθ::Cholesky{Float64,Array{Float64, 2}},
    choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}, expr_mid, Σθ_prime, X::Array{Float64,2}, gλz, Σθ_inv_X, Σθ_inv_y)
    AA = choleskyXΣX\(expr_mid)*(choleskyXΣX\(X'*(Σθ_inv_y)))
    BB = - (choleskyXΣX\(X'*(choleskyΣθ\(Σθ_prime*(Σθ_inv_y)))))
    βhat_prime_theta = AA + BB
    βhat_prime_theta = reshape(βhat_prime_theta, size(βhat_prime_theta, 1), size(βhat_prime_theta, 2)) #turn 1D array into 2D array
end

"""
First derivative of qtilde with respect to theta
"""
function compute_qtilde_prime_theta(gλz, X::Array{Float64,2}, βhat, βhat_prime_theta,
    choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, Σθ_prime::Array{Float64,2}, Σθ_inv_X, Σθ_inv_y)
    meanvv = gλz - X*βhat
    rr = X * βhat_prime_theta
    AA = (-rr)' * (Σθ_inv_y - Σθ_inv_X * βhat)  #(choleskyΣθ \ meanvv)
    BB = - meanvv' * (choleskyΣθ \ (Σθ_prime * (Σθ_inv_y - Σθ_inv_X * βhat)))
    CC =  meanvv' * ( - Σθ_inv_X * βhat_prime_theta)  #(choleskyΣθ \ (-rr))
    qtilde_prime_theta = AA .+ BB .+ CC
end

"""
Compute θ derivatives of log likelihood

-0.5 * logdet(XΣX) - 0.5 * logdet(Σ) - (n-p)/2 * log(qtilde) + (1-p/n) * log(J)

(first 3 terms)
"""
function btg_derivatives_per_θ(choleskyΣθ, choleskyXΣX, Σθ_prime, Σθ_inv_X, g_of_y, βhat, qtilde, Fx, Σθ_inv_y)
    Σθ_inv_Σθ_prime = choleskyΣθ\Σθ_prime
    XΣX_prime = - Fx'*(Σθ_inv_Σθ_prime)*(Σθ_inv_X)
    βhat_prime_theta = compute_betahat_prime_theta(choleskyΣθ, choleskyXΣX, -XΣX_prime, Σθ_prime, Fx, g_of_y, Σθ_inv_X, Σθ_inv_y)
    qtilde_prime_theta = compute_qtilde_prime_theta(g_of_y, Fx, βhat, βhat_prime_theta, choleskyΣθ, Σθ_prime, Σθ_inv_X, Σθ_inv_y)
    logdetΣθ_prime_θ = tr(Σθ_inv_Σθ_prime)
    logdetXΣX_prime_θ = tr(choleskyXΣX \ XΣX_prime)
    log_qtilde_prime_theta = qtilde_prime_theta/qtilde
    return (log_qtilde_prime_theta, logdetΣθ_prime_θ, logdetXΣX_prime_θ)
end
