include("../../../src/BTG.jl")

#@testset "test construct" begin
train = get_random_training_data()
w = construct(CovarianceMatrixCache(), Dict("θ"=>3.0, "var" => 1.0, "noise_var"=>2.0), train, RBFKernelModule())
@test w.kernel_module.var == 1.0
@test w.kernel_module.θ == 3.0
@test w.kernel_module.noise_var == 2.0
@test w.chol == cholesky(KernelMatrix(RBFKernelModule(Dict("θ"=>3.0, "var" => 1.0, "noise_var"=>2.0)), getposition(train)))

b = BufferDict()
cache = lookup_or_compute(b, "hi", CovarianceMatrixCache(), Dict("θ"=>3.0, "var" => 1.0, "noise_var"=>2.0),
          train, RBFKernelModule(); lookup = true, derivative = true)
cache
#end
