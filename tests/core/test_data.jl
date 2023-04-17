using Test
include("../../src/core/data.jl")

#%%
#@testset "training_data" begin
x = rand(100, 10)
Fx = rand(100, 3)
y = rand(100)
td = TrainingData(x, Fx, y)
@test getposition(td) == x
@test getcovariate(td) == Fx
@test getlabel(td) == y
@test getdimension(td) == size(x, 2)
@test getcovariatedimension(td) == size(Fx, 2)
@test getnumpoints(td) == size(x, 1)
x′ = rand(10)
fx′ = rand(3);
y′ = rand();
td′ = update!(td, x′, fx′, y′)
@test getposition(td′)[1:end-1, :] == x
@test getposition(td′)[end, :] == x′
@test getcovariate(td′)[1:end-1, :] == Fx
@test getcovariate(td′)[end, :] == fx′
@test getlabel(td′)[1:end-1] == y
@test getlabel(td′)[end] == y′ / getmaxlabel(td′)
@test getnumpoints(td′) == size(x, 1) + 1
#end

#%%

@testset "testing_data" begin
    x0 = rand(100, 10)
    Fx0 = rand(100, 3)
    y0 = rand(100)
    td = TestingData(x0, Fx0; y0_true = y0)
    @test getposition(td) == x0
    @test getcovariate(td) == Fx0
    @test getlabel(td) == y0
    @test getdimension(td) == size(x0, 2)
    @test getcovariatedimension(td) == size(Fx0, 2)
    @test getnumpoints(td) == size(x0, 1)
    @test getnumpoints(td) == size(x0, 1)
end
