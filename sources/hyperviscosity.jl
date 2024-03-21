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
    hv_differentiation_matrix = sum(hv_differentiation_matrices)

    # Scale hv by gamma 
    gamma = c * domain.pd.dx_min^(2 * k)

    return (; hv_differentiation_matrix, gamma, c)
end

function (source::SourceHyperviscosityFlyer)(du, u, t, domain, equations,
                                             solver::PointCloudSolver, semi_cache)
    basis = solver.basis
    pd = domain.pd
    @unpack hv_differentiation_matrix, gamma = cache
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded = semi_cache

    # Compute the hyperviscous dissipation
    # flux_values = local_values_threaded[1] # operator directly on u and du
    apply_to_each_field(mul_by_accum!(hv_differentiation_matrix,
                                      gamma),
                        du, u)
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
    # rbf_differentiation_matrices = compute_flux_operator(solver, domain)
    # Can just reuse rbf_differentiation_matrices in update

    # Containers for eps_uw and eps_rv
    nvars = nvariables(equations)
    uEltype = real(solver)
    eps_uw = zeros(uEltype, pd.num_points)
    eps_rv = zeros(uEltype, pd.num_points)
    eps = zeros(uEltype, pd.num_points)
    # eps_uw = allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)
    # eps_rv = allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)
    # eps = allocate_nested_array(uEltype, nvars, (pd.num_points,), solver)

    return (; eps_uw, eps_rv, eps, time_history, sol_history)
end

function update_upwind_visc!(eps_uw, u,
                             equations::CompressibleEulerEquations2D, domain, cache)
    gamma = equations.gamma
    set_to_zero!(eps_uw)

    for idx in eachindex(eps_uw)
        # Convert from conservative to primitive variables
        rho, v1, v2, p = cons2prim(u[idx], equations)

        # Compute local speed (magnitude of velocity) and sound speed
        speed = sqrt(v1^2 + v2^2)
        sound_speed = sqrt(gamma * p / rho)

        # h_loc is minimum pairwise distance between points in a patch centered
        # around x_i where patch consists of 5 points closest to x_i
        # instead we just take the distance from x_i to the nearest neighbor
        h_loc = norm(domain.pd.points[idx] - domain.pd.points[domain.pd.neighbors[idx][2]])

        # Calculate upwind viscosity for the current point
        eps_uw[idx] = 0.5 * h_loc * (speed + sound_speed)  # Assuming h_loc is uniform; adjust as needed
    end
end

function (source::SourceResidualViscosityTominec)(du, u, t, domain, equations,
                                                  solver::PointCloudSolver, semi_cache)
    basis = solver.basis
    pd = domain.pd
    @unpack eps_uw, eps_rv, eps, time_history, sol_history = cache
    @unpack rbf_differentiation_matrices, u_values, local_values_threaded, rhs_local_threaded = semi_cache

    # Update eps
    update_upwind_visc!(eps_uw, u, equations, domain, semi_cache)
    # update_residual_visc!(eps_rv, u, u_values, local_values_threaded,
    #                       rbf_differentiation_matrices)
    # update_eps!(eps, eps_uw, eps_rv)

    # Compute the hyperviscous dissipation
    # We need to apply P ⋅ u = Dx' diag(eps) Dx u + Dy' diag(eps) Dy u + ...
    # Handle one dim at a time, then accumulate into du
    local_u = local_values_threaded[1]
    local_rhs = rhs_local_threaded[1]
    set_to_zero!(local_u)
    set_to_zero!(local_rhs)
    for j in eachdim(domain)
        apply_to_each_field(mul_by!(rbf_differentiation_matrices[j]),
                            local_u, u)
        apply_to_each_field((out, x) -> out .= x .* eps, local_u)
        apply_to_each_field(mul_by_accum!(rbf_differentiation_matrices[j]', 1),
                            local_rhs, local_u)
    end
    du .+= local_rhs
end