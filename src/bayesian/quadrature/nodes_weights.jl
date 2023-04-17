using Distributions
using Sobol
using DelimitedFiles


struct nodesWeights
    nodes::Array{Float64, 2}
    weights::Array{Float64, 1}
end

"""
Computes quadrature nodes and weights
Returns:
    - Array of size n x (d+1), where last column is weights
"""
function get_nodes_weights(q::QuadType, range::Array{T, 2} where T<:Real;
                           levels::Union{Int, Nothing} = nothing, num_nodes::Int = 10,
                           verbose=false)
    dim = size(range)[1]
    if q == QuadType(0) #Gaussian
        NW = Array{Float64, 2}(undef, num_nodes^(dim), dim+1)
        nodes, weights = gausslegendre(num_nodes)
        # iterate over each dimension and do transformation
        N = Array{Float64, 2}(undef, num_nodes, dim)
        W = Array{Float64, 2}(undef, num_nodes, dim)
        for d in 1:dim
            nodes_d, weights_d = affineTransform(nodes, weights, reshape(range[d, :], 1, 2))
            N[:, d] .= nodes_d
            W[:, d] .= weights_d
        end
        # reshape N, W, put into NW
        for (i, (p, w)) in enumerate(zip(Iterators.product([N[:, i] for i in 1:dim]...), Iterators.product([W[:, i] for i in 1:dim]...)))
            NW[i, 1:dim] .= p
            NW[i, end] = reduce(*, w)
        end
        return NW

    elseif q == QuadType(1) # MonteCarlo
        NW = Array{Float64, 2}(undef, num_nodes, dim+1)
        NW[:, end] .= reduce(*, [(range[i, 2] - range[i, 1]) for i in 1:dim])/num_nodes
        for i = 1:dim
            NW[:, i] .= rand(Distributions.Uniform(range[i, 1], range[i, 2]), num_nodes)
        end
        return NW

    elseif q == QuadType(2) # QuasiMonteCarlo
        if verbose
            print("\nUsing QMC, dimension = $(dim), num_nodes = $(num_nodes)")
        end
        NW = Array{Float64, 2}(undef, num_nodes, dim+1)
        NW[:, end] .= reduce(*, [(range[i, 2] - range[i, 1]) for i in 1:dim])/num_nodes
        s = SobolSeq(range[:,1], range[:,2])
        NW[:, 1:dim] = hcat([next!(s) for i = 1:num_nodes]...)'
        return NW

    elseif q == QuadType(3) #SparseGrid
        levels = levels == nothing ? (dim < 7 ? 10-dim : 4) : levels
        #path = "src/bayesian/quadrature/quadratureData/GQU/GQU_d$(dim)_l$(levels).asc"
        path = joinpath(abspath(@__DIR__), "quadratureData/GQU/GQU_d$(dim)_l$(levels).asc")
        grids = readdlm(path, ',', Float64)
        #grids = readdlm(abspath("src/bayesian/quadrature/quadratureData/GQU/GQU_d$(dim)_l$(levels).asc"), ',', Float64)
        num_nodes = size(grids, 1)
        NW = Array{Float64, 2}(undef, num_nodes, dim+1)
        w_temp = grids[:, end]
        for d in 1:dim
            NW[:, d], w_temp_new = affineTransform(grids[:, d], w_temp, reshape(range[d, :], 1, 2), r_orig = [0,1])
            w_temp = w_temp_new
        end
        NW[:, end] .= w_temp
        if verbose
            print("\nUsing Sparse grid, dimension = $(dim), level = $(levels), $(num_nodes) quadrature nodes.")
        end
        return NW
    else
        error("Quad type not recognized: $(q)")
    end

end
