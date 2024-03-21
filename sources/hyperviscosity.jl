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
function SourceHyperviscosityFlyer(solver, equations, domain; k = 2, c = 1.0)
    cache = (; create_flyer_hv_cache(solver, equations, domain, k, c)...)

    SourceHyperviscosityFlyer{typeof(cache)}(cache)
end

function create_flyer_hv_cache(solver::PointCloudSolver, equations,
                               domain::PointCloudDomain, k::Int, c::Real)
    # Get basis and domain info
    basis = solver.basis
    pd = domain.pd

    # Create the actual operators
    # hv_differentiation_matrices operator
    # k is order of Laplacian, actual div order is 2k
    hv_differentiation_matrices = compute_flux_operator(solver, domain, 2 * k)

    # Scale hv by gamma 
    gamma = c * domain.pd.dx_min^(2 * k)

    return (; hv_differentiation_matrices, gamma, c)
end

function (source::SourceHyperviscosityFlyer)(du, u, t, domain, equations,
                                             solver::PointCloudSolver, semi_cache)
    basis = solver.basis
    pd = domain.pd
    hv_differentiation_matrices = cache.hv_differentiation_matrices
    gamma = cache.gamma
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded = semi_cache

    # Compute the hyperviscous dissipation
    # flux_values = local_values_threaded[1] # operator directly on u and du
    for i in eachdim(domain)
        for j in eachdim(domain)
            apply_to_each_field(mul_by_accum!(hv_differentiation_matrices[j],
                                              gamma),
                                du, u)
        end
    end
end

"""
    SourceHyperviscosityTominec

A struct containing everything needed to describe a hyperviscous
dissipation term for an RBF-FD discretization. 
## References

- Tominec (2023)
  Residual Viscosity Stabilized RBF-FD Methods for Solving
  Nonlinear Conservation Laws
  [doi: 10.1007/s10915-022-02055-8](https://doi.org/10.1007/s10915-022-02055-8)
Tominec computes Δᵏ operator from Δ'*Δ. Results in more fill in
but usually more stable than Flyer.
"""
struct SourceHyperviscosityTominec{Cache}
    cache::Cache

    function SourceHyperviscosityTominec{Cache}(cache::Cache) where {
                                                                     Cache}
        new(cache)
    end
end

"""
    SourceHyperviscosityTominec(solver, equations, domain)

Construct a hyperviscosity source for an RBF-FD discretization.
Designed for k=2
"""
function SourceHyperviscosityTominec(solver, equations, domain; c = 1.0)
    cache = (; create_tominec_hv_cache(solver, equations, domain, c)...)

    SourceHyperviscosityTominec{typeof(cache)}(cache)
end

function create_tominec_hv_cache(solver::PointCloudSolver, equations,
                                 domain::PointCloudDomain, c::Real)
    # Get basis and domain info
    basis = solver.basis
    pd = domain.pd

    # Create the actual operators
    # hv_differentiation_matrices operator
    # k is order of Laplacian, actual div order is 2k
    initial_differentiation_matrices = compute_flux_operator(solver, domain, 2)
    lap = sum(initial_differentiation_matrices)
    # dxx_dyy = initial_differentiation_matrices[1] + initial_differentiation_matrices[2]
    hv_differentiation_matrix = lap' * lap

    # Scale hv by gamma 
    gamma = c * domain.pd.dx_min^(2 * 2 + 0.5)

    return (; hv_differentiation_matrix, gamma, c)
end

function (source::SourceHyperviscosityTominec)(du, u, t, domain, equations,
                                               solver::PointCloudSolver, semi_cache)
    basis = solver.basis
    pd = domain.pd
    hv_differentiation_matrix = cache.hv_differentiation_matrix
    gamma = cache.gamma
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded = semi_cache

    # Compute the hyperviscous dissipation
    # flux_values = local_values_threaded[1] # operator directly on u and du
    apply_to_each_field(mul_by_accum!(hv_differentiation_matrix,
                                      gamma),
                        du, u)
end

"""
    SourceHyperviscosityTominec

A struct containing everything needed to describe a targeted
dissipation term for an RBF-FD discretization. 
## References

- Tominec (2023)
  Residual Viscosity Stabilized RBF-FD Methods for Solving
  Nonlinear Conservation Laws
  [doi: 10.1007/s10915-022-02055-8](https://doi.org/10.1007/s10915-022-02055-8)
"""
struct SourceResidualViscosityTominec{Cache}
    cache::Cache

    function SourceResidualViscosityTominec{Cache}(cache::Cache) where {
                                                                        Cache}
        new(cache)
    end
end

"""
SourceResidualViscosityTominec(solver, equations, domain)

Construct a targeted Residual Viscosity source for an RBF-FD discretization.
Designed for k=2
"""
function SourceResidualViscosityTominec(solver, equations, domain; c = 1.0)
    cache = (; create_tominec_rv_cache(solver, equations, domain, c)...)

    SourceResidualViscosityTominec{typeof(cache)}(cache)
end

function create_tominec_rv_cache(solver::PointCloudSolver, equations,
                                 domain::PointCloudDomain, c::Real)
    # Get basis and domain info
    basis = solver.basis
    pd = domain.pd

    # Create the actual operators
    # hv_differentiation_matrices operator
    # k is order of Laplacian, actual div order is 2k
    initial_differentiation_matrices = compute_flux_operator(solver, domain, 2)
    lap = sum(initial_differentiation_matrices)
    # dxx_dyy = initial_differentiation_matrices[1] + initial_differentiation_matrices[2]
    hv_differentiation_matrix = lap' * lap

    # Scale hv by gamma 
    gamma = c * domain.pd.dx_min^(2 * 2 + 0.5)

    return (; hv_differentiation_matrix, gamma, c)
end

function (source::SourceResidualViscosityTominec)(du, u, t, domain, equations,
                                                  solver::PointCloudSolver, semi_cache)
    basis = solver.basis
    pd = domain.pd
    hv_differentiation_matrix = cache.hv_differentiation_matrix
    gamma = cache.gamma
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded = semi_cache

    # Compute the hyperviscous dissipation
    # flux_values = local_values_threaded[1] # operator directly on u and du
    apply_to_each_field(mul_by_accum!(hv_differentiation_matrix,
                                      gamma),
                        du, u)
end