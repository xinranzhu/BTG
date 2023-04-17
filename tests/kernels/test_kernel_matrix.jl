include("../../src/kernels/kernel_module.jl")
include("../../src/kernels/rbf.jl")
include("../../src/kernels/kernel_matrix.jl")

#%% Single dimension

@testset "1" begin
    km = RBFKernelModule(Dict("θ" => 1.5, "var" => 1.0, "noise_var"=>0.0))

    x = reshape([1; 2.0], 2, 1)
    size(x, 1)
    evaluate(km, x[1, :], x[2, :]; θ = 1.0)

    @test KernelMatrix(km, x) == KernelMatrix(km, x; θ = 1.5);
    @test KernelMatrix(km, x) != KernelMatrix(km, x; θ = 1.54);
end

#%% multi-dimension
@testset "2" begin
    km = RBFKernelModule(Dict("θ" =>  [0.5, 1.2], "var" => 1.0, "noise_var"=>0.0))
    x = reshape([1 1.6; 2.0 2.4], 2, 2)
    @test KernelMatrix(km, x; θ = [0.5, 1.2]) == KernelMatrix(km, x)
    @test KernelMatrix(km, x; θ = [0.4, 1.2]) != KernelMatrix(km, x)
    @test KernelMatrix(km, x; θ = [0.5, 1.2], var = 1.0) == KernelMatrix(km, x)
    @test KernelMatrix(km, x; θ = [0.5, 1.2], var = 2.02) != KernelMatrix(km, x)
end
