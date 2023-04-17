"""
removes brackets around arrays of length 1 in values of d
"""
function remove_brackets!(d)
        for k in keys(d)
                if length(d[k]) == 1
                        d[k] = d[k][1]
                end
        end
        return d
end
