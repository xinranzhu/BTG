struct BayesianModule
    domain::AbstractQuadratureDomain
    buffer_integrator_pair::BufferIntegratorPair
    """
    Initialize domain using QuadratureSpecs
    """
    function bayesian_module(specs::QuadratureSpecs,
        buffer_integrator_pair::Pair{Function, Dict{String, AbstractBuffer}}
        )
        qd = QuadratureDomain(specs)

        return new(qd, )
    end
    function BayesianModule()
        return new()
    end
end
