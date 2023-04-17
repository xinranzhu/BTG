module linear_algebra_utils
    using Revise
    using LinearAlgebra
    #random PSD matrix
    function get_rand_psd(n)
        M = rand(n, n)
        return M'*M
    end
    export get_rand_psd

    function diag(m::Array{T} where T)
        @assert size(m, 1)==size(m, 2)
        x = zeros(size(m, 1))
        for i = 1:size(m, 1)
            x[i] = m[i, i]
        end
        return x
    end
end
