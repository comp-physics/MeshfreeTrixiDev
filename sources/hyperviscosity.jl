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
function SourceHyperviscosityFlyer(solver, equations, domain; k = 4)
    cache = (; create_flyer_hv_cache(solver, equations, domain, k)...)

    SourceHyperviscosityFlyer{typeof(cache)}(cache)
end

function create_flyer_hv_cache(solver::PointCloudSolver, equations,
                               domain::PointCloudDomain, k::Int)
    # Get basis and domain info
    basis = solver.basis
    pd = domain.pd

    # Create the actual operators
    # hv_differentiation_matrices operator
    hv_differentiation_matrices = compute_flux_operator(solver, domain, k)

    return (; hv_differentiation_matrices)
end

function (source::SourceHyperviscosityFlyer)(du, u, t, domain, equations,
                                             solver::PointCloudSolver, semi_cache)
    basis = solver.basis
    pd = domain.pd
    hv_differentiation_matrices = cache.hv_differentiation_matrices
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded = semi_cache

    # Compute the hyperviscous dissipation
    flux_values = local_values_threaded[1]
    for i in eachdim(domain)
        for e in eachelement(domain, solver, semi_cache)
            flux_values[e] = flux(u_values[e], i, equations)
            for j in eachdim(domain)
                apply_to_each_field(mul_by_accum!(hv_differentiation_matrices[j],
                                                  1),
                                    du, flux_values)
            end
        end
    end
end