using Test
include("../src/BTG_simulation/mock_btg.jl")
using .mock_btg
import .mock_btg.train_buffer
import .mock_btg.test_buffer
import .mock_btg.MockBTG

@testset "instantiation" begin
    train = mock_btg.train_buffer(ones(5000, 100))
    model = ax = mock_btg.MockBTG(train)
    @test model.TrainBuffer == train
end
