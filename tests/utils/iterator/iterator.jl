using Test
# using Plots
include("../../../src/utils/iterator/iterator.jl")


it = ((i, j) for (i, j) in [(100, 0.05), (1, 0.05), (3, .1), (5, .2), (6, 0.3), (7, 0.3)])


sub_it, frac, sum = get_iter_dominant_weights(it; threshold = 0.85)

total = 0.0
for (n, w) in sub_it
    #global total = total + w
end
@test frac ≈ 0.6666666 atol = 1e-5
@test sum ≈ 0.9 atol = 1e-5
