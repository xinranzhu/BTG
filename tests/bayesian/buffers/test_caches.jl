include("../../../src/BTG.jl")
using Test


#%% WGP caches
b = BufferDict()
dict = Dict("θ"=>0.8, "var"=>1.2, "noise_var"=>0.001, "x0"=>reshape([1.0], 1, 1))
kernel = RBFKernelModule()
ntrain = 5; dimx = 1
trainingdata0 = get_random_training_data(;n = ntrain, d = dimx)

res = construct(TestingCache(), dict, trainingdata0,
   kernel, "covar_buf_name", b;
   derivatives = true)
@test prod(size(res.Bθ_prime_x))== 5
@test prod(size(res.Eθ_prime_x))== 1
@test prod(size(res.Dθ_prime_x))== 1

## Transform cache

b = BufferDict()
my_transform = SinhArcSinh(1.0, 1.0)
tdict = Dict("a"=>1.2, "b"=>2.3)
y = [1.0, 2.0, 3.0]

res = construct(TransformCache(), tdict, my_transform, y)

res_lookup = lookup_or_compute(b, "transform_buffer", TransformCache(), tdict, my_transform, y)

@test res.g_of_y == res_lookup.g_of_y


#%% BTG Caches

@testset "Test construct function of BTGCovarianceCache, single-dim trainingdata" begin
    b = BufferDict()
    dict = Dict("θ"=>0.8, "var"=>1.2, "noise_var"=>0.001)
    kernel = RBFKernelModule()
    ntrain = 5; dimx = 1
    trainingdata0 = get_random_training_data(;n = ntrain, d = dimx)

    new_cache = lookup_or_compute(b, "BTGCovarianceCache", BTGCovarianceCache(), dict,
    trainingdata0, kernel; lookup = true, derivative = true, store = true)

    kernel_true = RBFKernelModule(dict)
    kernel_matrix_true = Symmetric(KernelMatrix(kernel_true, getposition(trainingdata0), jitter = 1e-10))
    choleskyΣθ_true = cholesky(kernel_matrix_true)
    dΣθ_true  = evaluate_derivative_θ(kernel_true, getposition(trainingdata0), getposition(trainingdata0), jitter = 1e-10)
    Fx = getcovariate(trainingdata0)
    Σθ_inv_X_true = choleskyΣθ_true\Fx
    XΣX_true = Fx'*Σθ_inv_X_true
    choleskyXΣX_true = cholesky(Symmetric(XΣX_true))
    logdetΣθ_true = logdet(choleskyΣθ_true)
    logdetXΣX_true = logdet(XΣX_true)


    # @test new_cache.kernel_module == kernel_true
    @test typeof(new_cache) == BTGCovarianceCache
    @test norm(new_cache.Σθ_inv_X .- Σθ_inv_X_true) < 1e-3
    @test norm(new_cache.choleskyΣθ.L .- choleskyΣθ_true.L) < 1e-3
    @test norm(new_cache.choleskyXΣX.L .- choleskyXΣX_true.L) < 1e-3
    @test abs(new_cache.logdetΣθ - logdetΣθ_true) < 1e-3
    @test abs(new_cache.logdetXΣX - logdetXΣX_true) < 1e-3
    @test norm(new_cache.dΣθ .- dΣθ_true) < 1e-3
    @test new_cache.n == ntrain

end



@testset "Test BTGCovarianceCache and TrainigCache, multi-dim trainingdata" begin
    b = BufferDict()
    buffer = Buffer()
    ntrain = 5; dimx = 2
    trainingdata0 = get_random_training_data(;n = ntrain, d = dimx)
    xtrain = getposition(trainingdata0)
    ytrain = getlabel(trainingdata0)
    Fxtrain = getcovariate(trainingdata0)

    dict = Dict("θ"=>[0.8, 0.76], "var"=>1.2, "noise_var"=>0.001, "a"=>1.1, "b"=>0.3)
    kernel = RBFKernelModule()
    transform = SinhArcSinh()
    input_dict = build_input_dict(transform, dict)
    input_dict_kernel = build_input_dict(kernel, dict)

    g(y) = evaluate(transform, input_dict, y)
    g_of_y = g.(ytrain)

    ## test BTGCovarianceCache
    new_cache = lookup_or_compute(b, "newBTGCovarianceCache", BTGCovarianceCache(), input_dict_kernel,
                                trainingdata0, kernel; lookup = true, derivative = true, store = true)
    # true values
    kernel_true = RBFKernelModule(input_dict_kernel)
    kernel_matrix_true = Symmetric(KernelMatrix(kernel_true, xtrain, jitter = 1e-10))
    choleskyΣθ_true = cholesky(kernel_matrix_true)
    dΣθ_true  = evaluate_derivative_θ(kernel_true, xtrain, xtrain, jitter = 1e-10)
    Σθ_inv_X_true = choleskyΣθ_true\Fxtrain
    XΣX_true = Fxtrain'*Σθ_inv_X_true
    choleskyXΣX_true = cholesky(Symmetric(XΣX_true))
    logdetΣθ_true = logdet(choleskyΣθ_true)
    logdetXΣX_true = logdet(XΣX_true)
    #test each
    @test typeof(new_cache) == BTGCovarianceCache
    @test norm(new_cache.Σθ_inv_X .- Σθ_inv_X_true) < 1e-3
    @test norm(new_cache.choleskyΣθ.L .- choleskyΣθ_true.L) < 1e-3
    @test norm(new_cache.choleskyXΣX.L .- choleskyXΣX_true.L) < 1e-3
    @test abs(new_cache.logdetΣθ - logdetΣθ_true) < 1e-3
    @test abs(new_cache.logdetXΣX - logdetXΣX_true) < 1e-3
    @test norm(new_cache.dΣθ .- dΣθ_true) < 1e-3
    @test new_cache.n == ntrain


    ## test training cache
    new_training_cache = lookup_or_compute(b, "BTGTrainingCache", BTGTrainingCache(), dict,
    trainingdata0, kernel, g_of_y, "newBTGCovarianceCache"; lookup = true)

    Σθ_inv_y_true = choleskyΣθ_true\g_of_y
    βhat_true = choleskyXΣX_true\(Fxtrain'*Σθ_inv_y_true)
    qtilde_true = g_of_y'*Σθ_inv_y_true  - 2*g_of_y'*Σθ_inv_X_true*βhat_true + βhat_true'*Fxtrain'*Σθ_inv_X_true*βhat_true

    @test norm(new_training_cache.βhat .- βhat_true) < 1e-3
    @test abs(new_training_cache.qtilde - qtilde_true) < 1e-3
    @test norm(new_training_cache.Σθ_inv_y - Σθ_inv_y_true) < 1e-3

    ## test testing cache
    x0 = rand(1, 2)
    Fx0 = ones(1, 1)
    input_dict_test = deepcopy(input_dict_kernel)
    input_dict_test["x0"] = x0 # add x0 and Fx0 to input_dict_kernel
    input_dict_test["Fx0"] = Fx0
    new_testing_cache = lookup_or_compute(buffer, b, BTGTestingCache(), input_dict_test,
                                          trainingdata0, kernel, "newBTGCovarianceCache"; lookup = true)

    Eθ_true = KernelMatrix(kernel_true, x0; noise_var = 0.0, jitter = 1e-10)
    Bθ_true = KernelMatrix(kernel_true, x0, xtrain, noise_var = 0.0)
    ΣθinvBθ_true = choleskyΣθ_true\Bθ_true'
    Dθ_true = Eθ_true - Bθ_true*(choleskyΣθ_true\Bθ_true') .+ input_dict_kernel["noise_var"][1] .+  1e-10
    Hθ_true = Fx0 - Bθ_true*Σθ_inv_X_true
    Cθ_true = Dθ_true + Hθ_true*(choleskyXΣX_true\Hθ_true')

    @test norm(Eθ_true .- new_testing_cache.Eθ) < 1e-3
    @test norm(Bθ_true .- new_testing_cache.Bθ) < 1e-3
    @test norm(ΣθinvBθ_true .- new_testing_cache.ΣθinvBθ) < 1e-3
    @test norm(Dθ_true .- new_testing_cache.Dθ) < 1e-3
    @test norm(Hθ_true .- new_testing_cache.Hθ) < 1e-3
    @test norm(Cθ_true .- new_testing_cache.Cθ) < 1e-3

end
