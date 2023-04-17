include("../../../../src/BTG.jl")
using Test

#(a, b, da, db) = WarpedConditionalPosteriorAcquisition(d, x0, Fx0, y0, td, transform, kernel_module, BufferDict();
#     lookup = true, test_buffer = Buffer(), quant=nothing)

 #f(x0) = WarpedConditionalPosteriorAcquisition(d, reshape([x0[1]], 1, 1), Fx0, y0, td, transform, kernel_module, BufferDict();
 #lookup = true, test_buffer = Buffer(), quant=nothing)[1]
 #df(x0) = WarpedConditionalPosteriorAcquisition(d, reshape([x0[1]], 1, 1), Fx0, y0, td, transform, kernel_module, BufferDict();
 #lookup = true, test_buffer = Buffer(), quant=nothing)[3]
 #g(x0) = WarpedConditionalPosteriorAcquisition(d, reshape([x0[1]], 1, 1), Fx0, y0, td, transform, kernel_module, BufferDict();
 #lookup = true, test_buffer = Buffer(), quant=nothing)[2]
 #dg(x0) = WarpedConditionalPosteriorAcquisition(d, reshape([x0[1]], 1, 1), Fx0, y0, td, transform, kernel_module, BufferDict();
 #lookup = true, test_buffer = Buffer(), quant=nothing)[4]
 #(_, _, _, r4) = checkDerivative(f, df, 1.5) # a(x) = m(x) + beta * sigma(x)
#@show r4
