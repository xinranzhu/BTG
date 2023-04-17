"""
Monotonic warping function From R^1 to R^1
Composed by elementary nonlinear transformations
Math:
    T = [g1, g2, g3]
    T(y) = g1(g2(g3(y)))
    dT(y) = dg1'(g2(g3(y))) * dg2'(g3(y)) * dg3'(y)
    invT(x) = invg3(invg2(invg1(x)))
"""
struct ComposedTransformation <:AbstractComposedTransform
    transform_list::Array{AbstractTransform}
    function flatten(transform_list)
        new_list = []
        for item in transform_list
            if typeof(item)<:ComposedTransformation
                new_list = cat(new_list, flatten(item), dims=1)
            else
                new_list = cat(new_list, item, dims=1)
            end
        end
        return new_list
    end
    #flattened list of composed functions
    ComposedTransformation(transform_list) = new(flatten(transform_list))
end

#d = {"COMPOSED_TRANSFORM_DICTS"=>[d1, d2, d3], "θ"=>1.0, "var"=>2.0}

"""
d is user-input dictionary containing values for all parameters, not just
those related to the composed transformation. This function extracts relevant
info from d and builds a completed composed transformation dict, possibly by
falling back on default values
Input examples:
d = Dict("var" => 1.0, "θ" => 0.03465,
         "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>0.99), Dict()])
d = Dict("var" => 1.0, "θ" => 0.03465,
        "COMPOSED_TRANSFORM_DICTS"=>[Dict(), Dict()])
Output examples:
my_composed_transform_dicts = [Dict("a"=>0.99, "b"=>default_b), Dict("a"=>default_a, "b"=>default_b)])

my_composed_transform_dicts= [Dict("a"=>default_a, "b"=>default_b), Dict("a"=>default_a, "b"=>default_b)])
"""
function build_input_dict(t::ComposedTransformation, d::Dict)::Array{Dict, 1}
    if !("COMPOSED_TRANSFORM_DICTS" in keys(d))
        @warn "COMPOSED_TRANSFORM_DICTS key not found in input dictionary"
        m = length(t.transform_list)
        return [transform.PARAMETER_DICT for transform in t.transform_list]
    end
    my_composed_transform_dicts = d["COMPOSED_TRANSFORM_DICTS"]
    transform_list = t.transform_list
    m = length(transform_list)
    for i = 1:m
        names = transform_list[i].names
        @assert issubset(keys(my_composed_transform_dicts[i]), names) #
    end
    for i = 1:m
        my_composed_transform_dicts[i] = convert(Dict{Any, Any}, my_composed_transform_dicts[i])
        tmp = build_input_dict(transform_list[i], my_composed_transform_dicts[i])
        my_composed_transform_dicts[i] = convert(Dict{Any, Any}, tmp)
    end
    return my_composed_transform_dicts
end


function evaluate(t::ComposedTransformation, array_dict, y::Real)
    transform_list_reversed = reverse(t.transform_list)
    array_dict_reversed = reverse(array_dict)
    out = evaluate(transform_list_reversed[1], array_dict_reversed[1], y) # g3(y)
    if length(transform_list_reversed) > 1
        for i in 2:length(transform_list_reversed)
            out = evaluate(transform_list_reversed[i], array_dict_reversed[i], out)
        end
    end
    return out
end


function evaluate_derivative_x(t::ComposedTransformation, array_dict, y::Real)
    transform_list_reversed = reverse(t.transform_list)
    array_dict_reversed = reverse(array_dict)
    out = evaluate_derivative_x(transform_list_reversed[1], array_dict_reversed[1], y)
    gy = evaluate(transform_list_reversed[1], array_dict_reversed[1], y)
    if length(transform_list_reversed) > 1
        for i in 2:length(transform_list_reversed)
            out = evaluate_derivative_x(transform_list_reversed[i], array_dict_reversed[i], gy) * out
            gy = evaluate(transform_list_reversed[i], array_dict_reversed[i], gy)
        end
    end
    return out
end

function evaluate_inverse(t::ComposedTransformation, array_dict, y::Real)
    transform_list = t.transform_list # T = [g1, g2, g3, ...]
    out = evaluate_inverse(transform_list[1], array_dict[1], y)
    if length(transform_list) > 1
        for i in 2:length(transform_list)
            out = evaluate_inverse(transform_list[i], array_dict[i], out)
        end
    end
    return out
end

"""
The array dict should contain complete transformation parameter key value pairs
"""
function evaluate_derivative_hyperparams(t::ComposedTransformation, array_dict, y::Real)
    transform_list = t.transform_list
    transform_list_reversed = reverse(transform_list)
    array_dict_reversed = reverse(array_dict)
    n_transform = length(transform_list_reversed)
    gy = y
    gys = [gy]  #y, g3(y), g2(g3(y))..
    for i = 1:n_transform
        gy = evaluate(transform_list_reversed[i], array_dict_reversed[i], gy)
        push!(gys, gy)
    end
    outer_nested_derivative = [] #ith  g1(g2(g_n-1'(x)))\circ g_n(y)....g1'(x)\circ g_2(g3....gn(y)), 1
    for i = n_transform-1:-1:1
        ct = ComposedTransformation(transform_list[1:i]) #g1 g2 g3
        deriv = evaluate_derivative_x(ct, array_dict[1:i], gys[n_transform - i + 1])
        push!(outer_nested_derivative, deriv)
    end
    push!(outer_nested_derivative, 1)
    inner_hyperparameter_derivative = [] #[Dict("a“=>gn'(a)(y), "b"=>gn'(b)(y)], [g_n-1'(a)(gn(y)), g_n-1'(b)(gn_(y))], ...]
    for i = 1:n_transform
        cur = evaluate_derivative_hyperparams(transform_list_reversed[i], array_dict_reversed[i], gys[i])
        push!(inner_hyperparameter_derivative, cur)
    end
    function augment(a::Real, b::Dict)
        for key in keys(b)
            b[key] = b[key]*a
        end
        return b
    end
    final_derivative = []
    for i = 1:n_transform
        b = inner_hyperparameter_derivative[i]
        a = outer_nested_derivative[i]
        push!(final_derivative, augment(a, b))
    end
    return reverse(final_derivative)
    # return final_derivative
end


function evaluate_derivative_x_hyperparams(t::ComposedTransformation, array_dict, y::Real)
    transform_list = t.transform_list
    n_transform = length(transform_list)
    transform_list_reversed = reverse(t.transform_list)
    array_dict_reversed = reverse(array_dict)
    gy = y
    gys = [gy]  #[y, g3(y), g2(g3(y)), g1(g2(g3(y)))] n+1
    for i = 1:n_transform
        gy = evaluate(transform_list_reversed[i], array_dict_reversed[i], gy)
        push!(gys, gy)
    end
    Ds = [] # [g1'(g2(g3(y))), g2'(g3(y)), g3'(y)]
    for i = 1:n_transform
        val = evaluate_derivative_x(transform_list[i], array_dict[i], gys[n_transform-i+1])
        push!(Ds, val)
    end; push!(Ds, 1.0)
    D2s = [] # [g1''(g2(g3(y))), g2''(g3(y)), g3''(y)]
    for i = 1:n_transform
        val2 = evaluate_derivative_x2(transform_list[i], array_dict[i], gys[n_transform-i+1])
        push!(D2s, val2)
    end
    array_dicts = Array{Array{Dict}}(undef, n_transform) #array of dictionaries of hyperparam derivatives of D1, ...Dn
    for i = 1:n_transform
        # D(g1'(g2(g3(y)))) - derivatives w.r.t g1 parameters only
        val1 = evaluate_derivative_x_hyperparams(transform_list[i], array_dict[i], gys[n_transform - i + 1])
        # D(g2(g3(y))) - derivatives w.r.t g2 g3 parameters
        val2 = evaluate_derivative_hyperparams(ComposedTransformation(transform_list[i+1:end]), array_dict[i+1:end], y) #array of dicts
        for l = 1:length(val2)
            for k in keys(val2[l])
                val2[l][k] = D2s[i] * val2[l][k]
            end
        end
        array_dicts[i] = cat(val1, val2, dims=1)
    end
    function add_combine!(D1, inc, names)
        for k in names
            D1[k] = k in keys(D1) ? D1[k] + inc[k] : inc[k]
        end
    end
    function augment(a::Real, b::Dict)
        w = deepcopy(b)
        for key in keys(w)
            w[key] = w[key]*a
        end
        return w
    end
    final_array_dicts = Array{Dict}(undef, n_transform) # [derivatives w.r.t g1 parameters, ...]
    for i = 1:n_transform
        final_dict = Dict()
        cur_array_dict = [array_dicts[j][i-j+1] for j = 1:i] #Derivatives of Ds w.r.t to some hyperparams
        cur_Ds = Ds[1:i]
        last_Ds = Ds[i+1:end]
        prod_last_Ds = prod(Ds[i+1:end])
        for j = 1:i
            cur_arr = deepcopy(cur_Ds)
            cur_arr[j] = 1.0
            add_combine!(final_dict, augment(prod(cur_arr), cur_array_dict[j]), transform_list[i].names)
        end
        final_array_dicts[i] = augment(prod_last_Ds, final_dict)
    end
    #return Ds, array_dicts
    return final_array_dicts
end


function plot_transform(t::ComposedTransformation, array_dict; 
                yrange=[0,100], figure_title=nothing, save_path=nothing)

    @assert length(yrange) == 2
    ygrid = collect(yrange[1]:0.01:yrange[2])
    g(y) = evaluate(t, array_dict, y)
    gygrid = g.(ygrid)
    
    PyPlot.close("all") #close existing windows
    PyPlot.plot(ygrid, gygrid)

    if figure_title != nothing
        PyPlot.title(figure_title, fontsize=10)
    end
    if save_path != nothing
        PyPlot.savefig("$(save_path).pdf")
        println("Figure saved: $(save_path).pdf")
    end

end