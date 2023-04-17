"""
Takes in

["θ", "var", "noise_var", [["a", "b"], ["a", "b"]]]
flattened([10.0, 10.0, 1.0, [[10.0, 10.0], [10.0, 10.0]]])

and outputs dictionary of the form

Dict("θ"=>10.0, "var"=>10.0, "noise_var"=>10.0, "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>10.0),
Dict("b"=>10.0)], [Dict("a"=>10.0), Dict("b")=>10.0])

Requires:
- nested array [[10.0, 10.0], [10.0, 10.0]] occurs only once
Other examples:

For multidimensional theta, takes in 
parameter_names = [["θ1", "θ2"], "var", [["a", "b"], ["c"]]]
Outputs:
    Dict("θ" => [0.8, 0.9], "var"=>0.1,             
            "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>10.0, "b"=>10.0), Dict("c"=>10.0)] )
"""
function flattened_list_to_dictionary(parameter_names, arr)
    # @assert count( x->typeof(x)<:Array, parameter_names)<=1
    dict = Dict()
    offset = 0
    # parameter_names = [["θ1", "θ2"], "var", [["a", "b"], ["c"]]]
    for i = 1:length(parameter_names)
        param = parameter_names[i] # [["a", "b", "c", "d"], ["a", "b"], ["a", "b"]], 
        if typeof(param)<:String
            dict[param] = arr[i+offset] #dictionary values must be arrays
        elseif typeof(param)<:Array && typeof(param[1])<:Array
            num_transform = length(param)
            @assert typeof(param[1])<:Array #nested array
            composed_arrays = Array{Dict{Any}, 1}(undef, num_transform)
            sizes = [length(param[k]) for k = 1:length(param)]
            total_sizes = sum(sizes)
            for k = 1:num_transform
                cur_d = Dict()
                for w = 1:sizes[k]
                    cur_d[param[k][w]] = arr[(i + w - 1 + offset + sum(sizes[1:k-1]))]
                end
                # push!(composed_arrays, cur_d)
                composed_arrays[k] = cur_d
            end
            dict["COMPOSED_TRANSFORM_DICTS"] = composed_arrays
            offset = offset + sum(sizes) - 1
        elseif typeof(param)<:Array && (typeof(param[1]) <: String) #["θ1", "θ2"...] 
            @assert startswith(param[1], "θ")
            dict["θ"] = convert(Array{Float64,1}, arr[(i+offset): (i+offset+length(param)-1)] )
            offset += length(param) - 1
        else
            error("Entries of parameter names must be string or array of strings (for multidimensional params) or array of array of strings (for composed transform)")
        end
    end
    return dict
end




function flatten_nested_list(arr)
    res = []
    if arr == []
        return []
    end
    for item in arr
        if typeof(item) <: Array
            res = cat(res, flatten_nested_list(item), dims=1)
        else
            push!(res, item)
        end
    end
    return res
end

# parameter_names =  ["var",  [["a", "b", "c", "d"], ["a", "b"]], ["θ1", "θ2"]]
# arr = [1., 2., 3. , 4., 5., 6., 7., 8., 9.]
# output: 
# final_list = [1., [[2., 3., 4., 5.], [6., 7.]], [8., 9.]]
function flattened_list_to_nested_list(parameter_names, arr)
    for item in arr
        @assert !(typeof(item)<:Array)
    end
    final_list = []
    counter = 1
    for i = 1:length(parameter_names)
        param = parameter_names[i]
        if typeof(param)<:Array && typeof(param[1])<:Array
            grouped_arr = []
            for inside_arr in parameter_names[i]
                my_inside_arr = []
                for _ in inside_arr
                    push!(my_inside_arr, arr[counter])
                    counter = counter + 1
                end
                push!(grouped_arr, my_inside_arr)
            end
            push!(final_list, grouped_arr)
        elseif typeof(param)<:Array && typeof(param[1])<:String # param =  ["θ1", "θ2"]
            grouped_arr = []
            for item in param
                push!(grouped_arr, arr[counter])
                counter += 1
            end
            push!(final_list, grouped_arr)
        else
            push!(final_list, arr[counter])
            counter = counter + 1
        end
    end
    return final_list
end

function nested_list_to_dictionary(parameter_names, arr)
    flattened_list = flatten_nested_list(arr)
    return flattened_list_to_dictionary(parameter_names, flattened_list)
end

"""
Inputs:
    d = Dict("θ" => [0.8, 0.9], "var"=>0.1,             
         "COMPOSED_TRANSFORM_DICTS"=>[Dict("a"=>10.0, "b"=>10.0), Dict("c"=>10.0)] )
    parameter_names =  ["var",  [["a", "b"], ["c"]], ["θ1", "θ2"]]
Outputs:
    [0.1, 10, 10, 10, 0.8, 0.9]
"""
function dictionary_to_flattened_list(parameter_names, d::Dict)
    arr = []
    for item in parameter_names
        if typeof(item)<:String
            push!(arr, d[item])
        elseif typeof(item)<:Array && typeof(item[1])<:Array
            inner_dict = d["COMPOSED_TRANSFORM_DICTS"]
            for j = 1:length(item)
                for k in item[j]
                    push!(arr, inner_dict[j][k])
                end
            end
        elseif typeof(item)<:Array && typeof(item[1])<:String # ["θ1", "θ2"]
            for theta_i in d["θ"]
                push!(arr, theta_i)
            end
        else
            error("type of items in parameter_names must be string or array")
        end
    end
    return arr
end

"""
Takes dictionary containing nested arrays and converts it to nested array form
"""
function dictionary_to_nested_list(parameter_names, d::Dict)
    arr = []
    for item in parameter_names
        if typeof(item)<:String
            push!(arr, d[item])
        elseif typeof(item)<:Array
            inner_dict = d["COMPOSED_TRANSFORM_DICTS"]
            grouped_arr = []
            for j = 1:length(item)
                cur_arr = []
                for k = 1:length(item[j])
                    push!(cur_arr, inner_dict[j][string(item[j][k])])
                end
                push!(grouped_arr, cur_arr)
            end
            push!(arr, grouped_arr)
        else
            error("type of items in parameter_names must be string or array")
        end
    end
    return arr
end
