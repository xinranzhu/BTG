using Test
#TODO Appveyor integration
#bayesian
include("bayesian/quadrature/test_quadrature_integration.jl")
include("bayesian/quadrature/test_quadrature_types.jl")
include("bayesian/quadrature/test_quadrature.jl")
include("bayesian/quadrature/test_domain.jl")
include("bayesian/buffers/test_buffers.jl")
include("bayesian/buffers/test_caches.jl")

#core
include("core/test_data.jl")
include("core/test_parameters.jl")
#kernel
include("kernels/test_kernel.jl")
#model
include("models/test_btg_obj.jl")
include("models/test_btg_optimize.jl")
include("models/test_btg_prediction.jl")
include("models/utils/test_list_dict_mapping.jl")
include("models/utils/test_parse_input.jl")
include("models/utils/test_quantile_bound.jl")
include("models/warped_model/test_likelihood.jl")
include("models/btg_model/test_btg_likelihood.jl")
include("models/btg_model/test_btg_conditional_posterior.jl")

#transforms
include("transforms/test_transforms.jl")
#utils
include("utils/derivative/test_derivative_checker.jl")

#btg simulation
include("BTG_simulation/bayesopt/warped_model/LCB.jl")
