include("../../../../src/BTG.jl")
using Test

@testset "test reformatter" begin
    ct1 = ComposedTransformation([Affine(1.0,2.0), SinhArcSinh(2.0, 3.0)])
    d = Dict("COMPOSED_TRANSFORM_PARAMETER_1"=>[1.9, 2.0], "COMPOSED_TRANSFORM_PARAMETER_2"=>[3.4, 5.2], "θ"=>3.4)
    rr = reformat(d, ct1)
    @test length(rr)==2
    @test rr["COMPOSED_TRANSFORM_DICTS"][1] == Dict("a"=>1.9, "b"=>2.0)
    @test rr["COMPOSED_TRANSFORM_DICTS"][2] == Dict("a"=>3.4, "b"=>5.2)
end
#%%

@testset "no effect on kernel params, but allocate space for Dict array" begin
    ct1 = ComposedTransformation([Affine(1.0,2.0), SinhArcSinh(2.0, 3.0)])
    d = Dict("θ"=>3.4,"var"=>3.5)
    rr = reformat(d, ct1)
    @show rr
    @test rr["θ"] == d["θ"] 
    @test rr["var"] == d["var"]
    @test prod(rr["COMPOSED_TRANSFORM_DICTS"] .== [Dict(), Dict()]) == 1

    ct1 = ComposedTransformation([Affine(1.0,2.0), SinhArcSinh(2.0, 3.0)])
    d = Dict("θ"=>3.4,"var"=>3.5, "COMPOSED_TRANSFORM_PARAMETER_1"=>[1.9, 2.0])
    rr = reformat(d, ct1)
    @show rr
    @test rr["θ"] == d["θ"] 
    @test rr["var"] == d["var"]
    @show rr["COMPOSED_TRANSFORM_DICTS"]
    @test prod(rr["COMPOSED_TRANSFORM_DICTS"] .== [Dict("a"=>1.9, "b"=>2.0), Dict()]) == 1

end

