"""
    SourceHyperviscosityFlyer

A struct containing everything needed to describe a hyperviscous
dissipation term for an RBF-FD discretization. 
## References

- Flyer (2016)
  Enhancing finite differences with radial basis functions:
  Experiments on the Navier-Stokes equations
  [doi: 10.1016/j.jcp.2016.02.078](https://doi.org/10.1016/j.jcp.2016.02.078)
Flyer implementation directly computes Δᵏ operator
"""
struct SourceHyperviscosityFlyer{Cache}
    cache::Cache

    function SourceHyperviscosityFlyer{Cache}(cache::Cache) where {
                                                                   Cache}
        new(cache)
    end
end

"""
    SourceHyperviscosityFlyer(solver, equations, domain)

Construct a hyperviscosity source for an RBF-FD discretization.
"""
function SourceHyperviscosityFlyer(solver, equations, domain)
    cache = (; create_flyer_hv_cache(solver, equations, domain)...)

    SourceHyperviscosityFlyer{typeof(cache)}(cache)
end

function create_flyer_hv_cache(solver::PointCloudSolver, equations,
                               domain::PointCloudDomain)
    # Get basis and domain info
    basis = solver.basis
    pd = domain.pd

    # Create the actual operators
    # hv_differentiation_matrices operator
    hv_differentiation_matrices = create_flyer_hv_operator(solver, equations, domain)

    return (; hv_differentiation_matrices)
end