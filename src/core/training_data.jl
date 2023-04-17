"""
Represents training dataset for regression problem
"""
mutable struct TrainingData <: AbstractTrainingData
    x::Array{Float64, 2} #matrix of horizontal location vectors stacked vertically
    Fx::Array{Float64, 2} #matrix of covariates
    y::Array{Float64, 1} #array of labels, normalized
    d::Int64 #dimension of location vectors in x
    p::Int64 #dimension of covariate vectors in Fx
    n:: Int64 # number of incorporated points
    x_mean::Array{Float64, 2} # mean values of x_train, 1 by dim
    x_std::Array{Float64, 2} # std of x_train, 1 by dim
    y_mean::Float64 # mean value of y_train, scalar
    y_std::Float64 # std of y_train, scalar
    function TrainingData(x, Fx, y; normalization=true)
        @assert Base.size(x, 1) == Base.size(Fx, 1)
        @assert Base.size(Fx, 1) == length(y)
        @assert typeof(x)<:Array{T} where T<:Float64 "x must be of type float"
        @assert typeof(Fx)<:Array{T} where T<:Float64 "Fx must be of type float"
        @assert typeof(y)<:Array{T} where T<:Float64 "y must be of type float"
        x_mean=mean(x, dims=1) # 1 by dim
        # x_mean = zeros(1, Base.size(x, 2))
        x_std=std(x, dims=1) # 1 by dim
        # x_std = ones(1, Base.size(x, 2))
        # y_mean=mean(y)
        # y_std=std(y)
        y_mean = 0
        # y_std = 1
        y_std = maximum(y)
        x_new = broadcast(-, x, x_mean)
        x_new = broadcast(/, x_new, x_std)
        y_new = broadcast(-, y, y_mean)
        y_new = broadcast(/, y_new, y_std)
        return new(x_new, Fx, y_new, Base.size(x, 2), Base.size(Fx, 2), Base.size(x, 1), x_mean, x_std, y_mean, y_std)
    end
end

"""
Delete ith training point to obtain new training data
"""
function leaveOut(t::TrainingData, i::Int64)
    n = t.n
    return TrainingData(t.x[[1:i-1; i+1:n], :], t.Fx[[1:i-1; i+1:n], :], t.y[[1:i-1; i+1:n]]; normalization = false)
end

"""
Extract ith data point as testingData
"""
function ith(t::TrainingData, i::Int64)
    return TestingData(t.x[i:i, :], t.Fx[i:i, :], y0_true = [t.y[i]])
end

function trainingdata(table; Xcols, Fxcols, ycol)
    x = Tables.matrix(table[:, Xcols])
    Fx = Tables.matrix(table[:, Fxcols])
    y = table[:, ycol]
    return TrainingData(x, Fx, y)
end

getposition(td::TrainingData) = td.x
getcovariate(td::TrainingData) = td.Fx
getlabel(td::TrainingData) = td.y
getdimension(td::TrainingData) = td.d
getcovariatedimension(td::TrainingData) = td.p
getnumpoints(td::TrainingData) = td.n

function get_original_x(td::TrainingData)
    x_new = broadcast(*, td.x, td.x_std)
    x_new = broadcast(+, x_new, td.x_mean)
    return x_new
end


function get_original_y(td::TrainingData)
    y_new = broadcast(*, td.y, td.y_std)
    y_new = broadcast(+, y_new, td.y_mean)
    return y_new
end




function update!(D::TrainingData, x, fx, y)
    x0 = broadcast(*, D.x, D.x_std)
    x0 = broadcast(+, x0, D.x_mean)
    y0 = broadcast(*, D.y, D.y_std)
    y0 = broadcast(+, y0, D.y_mean)
    X = [x0; x']
    Fx = [D.Fx; fx']
    Y = [y0; y]
    return TrainingData(X, Fx, Y)
end
