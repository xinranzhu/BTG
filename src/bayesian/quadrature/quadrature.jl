using IterTools
using FastGaussQuadrature
import IterTools: product, groupby
#misc
include("utils/utils.jl")
include("utils/composed_transformation_utils.jl")

include("../../../src/core/parameters.jl")
#main files
include("quadrature_types.jl")
include("nodes_weights.jl")
include("domain.jl")

include("utils/get_quadrature_domain.jl")
