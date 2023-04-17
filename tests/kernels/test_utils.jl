using Test
include("../../src/kernels/Utils/utils.jl")
import .utils.reshape

@testset "reshape" begin
    a = [3.14]
    @test reshape(a) == reshape([3.14], 1, 1)
    a = 3.14
    @test reshape(a) ==  a
    a = [1, 2, 3]
    @test reshape(a) == reshape(a, 3, 1)
    a = [1 2; 2 3; 3 4]
    @test reshape(a) == a
    a = nothing
    @test reshape(a) == a
end
