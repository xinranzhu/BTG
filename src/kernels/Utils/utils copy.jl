
module utils
using Distances
using LinearAlgebra
import Base: size, reshape

function reshape(a::Union{Array{T} where T, G where G<:Real, Nothing})
    if (a != nothing) && length(size(a)) == 1 #reshape into column vector, if not already a column vector
        a  = reshape(a, length(a), 1)
    end
    return a
end

end
