using Test
include("../../src/core/parameters.jl")
include("../../src/utils/test_utils.jl")

@testset "merge" begin
    parameters = [Parameter("a", 1, reshape([1, 2], 1, 2)), Parameter("b", 1, reshape([2, 3], 1, 2))]
    x = merge(parameters)
    @test x ≂ Parameter("MERGED_PARAMETER", 2, [1.0 2.0; 2.0 3.0])
end
