"""
Apply affine transformation to nodes to integration over range r (linear change of variables)
Input nodes are assumed to be tailored to [-1, 1]. Transformation of nodes gives integration nodes
for new domain [r[1], r[2]]
"""
function affineTransform(nodes::Array{A, 1} where A<:Real,
    weights::Array{B, 1} where B<:Real, r::Array{C, 2} where C<:Real; r_orig::Array{D, 1} where D<:Real = [-1, 1])
    center = (-r_orig[1]*r[2] + r_orig[2]*r[1]) / (r_orig[2] - r_orig[1])
    length = (r[2] - r[1]) / (r_orig[2] - r_orig[1])
    nodes = length .* nodes .+ center #shift and scale nodes
    weights = weights .* length #linarly scale weights
    return nodes, weights
end
