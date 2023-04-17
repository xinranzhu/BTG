mutable struct Counter
    f::Function
    count::Int64
end

Counter(f) = Counter(f, 0)

function (c::Counter)(x...; kwargs...)
    c.count += 1
    c.f(x...; kwargs...)
end
