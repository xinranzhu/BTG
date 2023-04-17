include("../../../src/BTG.jl")
using Test

x = TrainingData([1 2 ; 3.0 4.0], [ 1 2; 3.0 4.0], [1, 2.0])
sub_1 = leaveOut(x, 1)
sub_2 = leaveOut(x, 2)
@test sub_1.x == [3.0 4.0]
@test sub_2.x == [1.0 2.0]
@test sub_1.Fx == [3.0 4.0]
@test sub_2.Fx == [1.0 2.0]
@test sub_1.y == [1.0]
@test sub_2.y == [0.5]

t = ith(x, 1)
@test t.x0 == [1.0 2.0]
@test t.Fx0 == t.x0
t2 = ith(x, 2)
@test t2.x0 == [3.0 4.0]
@test t2.Fx0 == t2.x0
