"""
Represents testing data for regression problem. Currently supports single-point prediction.
"""
mutable struct TestingData<:AbstractTestingData
    x0::Array{T} where T<:Real
    Fx0::Array{T} where T<:Real
    y0_true::Union{Array{T}, Nothing} where T<:Real
    d::Int64
    p::Int64
    k::Int64
    TestingData(x0::Array{T, 2}, Fx0::Array{T, 2}; y0_true=nothing) where T<:Real = (@assert size(x0, 1)==size(Fx0, 1);  new(x0, Fx0, y0_true, size(x0, 2), size(Fx0, 2), size(x0, 1)))
    TestingData() = new()
end

function testdata(table; Xcols, Fxcols)
    x = Tables.matrix(table[:, Xcols])
    Fx = Tables.matrix(table[:, Fxcols])
    return TestingData(x, Fx, nothing)
end





getposition(td::TestingData) = td.x0
getcovariate(td::TestingData) = td.Fx0
getlabel(td::TestingData) = td.y0_true

getdimension(td::TestingData) = td.d
getcovariatedimension(td::TestingData) = td.p
getnumpoints(td::TestingData) = td.k
