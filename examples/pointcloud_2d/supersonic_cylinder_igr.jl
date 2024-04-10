using Revise
using MeshfreeTrixi
using OrdinaryDiffEq
using IterativeSolvers

# includet("../header.jl")

# Base Methods
approximation_order = 3
rbf_order = 3
# Specialized Methods
basis = PointCloudBasis(Point2D(), approximation_order;
                        approximation_type = RBF(PolyharmonicSpline(rbf_order)))
solver = PointCloudSolver(basis)

dir = "./medusa_point_clouds"
casename = "cyl_0_005"
domain_name = joinpath(dir, casename)
savename = casename * "_order_$approximation_order" * "_igr"
boundary_names = Dict(:inlet => 1, :outlet => 2, :bottom => 3, :top => 4, :cyl => 5)
domain = PointCloudDomain(solver, domain_name, boundary_names)
scatter(domain.pd.points, axis = (aspect = DataAspect(),))
for tag in keys(domain.boundary_tags)
    idx = domain.boundary_tags[tag].idx
    scatter!(domain.pd.points[idx], label = string(tag))
end

equations = CompressibleEulerEquations2D(1.4)
function initial_condition_cyl(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.4
    rho_v1 = 4.1
    rho_v2 = 0.0
    rho_e = 8.8
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
initial_condition = initial_condition_cyl
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => BoundaryConditionDoNothing(),
                       :top => boundary_condition_slip_wall,
                       :bottom => boundary_condition_slip_wall,
                       :cyl => boundary_condition_slip_wall)

# Test IGR
source_hv = SourceHyperviscosityTominec(solver, equations, domain;
                                        c = domain.pd.dx_min^(-2 + 0.5))
source_igr = SourceIGR(solver, equations, domain; alpha = 20 * domain.pd.dx_min^2,
                       linear_solver = gmres!)
sources = SourceTerms(hv = source_hv, igr = source_igr)
semi_cache = semi.cache
basis = solver.basis
pd = domain.pd
@unpack sigma, trace, trace_squared, alpha, lhs_operator = source_igr.cache
@unpack rbf_differentiation_matrices, u_values, local_values_threaded, rhs_local_threaded = semi_cache
du = deepcopy(ode.u0)
u = deepcopy(ode.u0)
MeshfreeTrixi.update_igr_rhs!(du, u, equations, domain, source_igr.cache, semi_cache)

# Solver
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)
tspan = (0.0, 4.0)
ode = semidiscretize(semi, tspan)

# Try sim
# summary_callback = SummaryCallback()
summary_callback = InfoCallback()
alive_callback = AliveCallback(alive_interval = 10)
# history_callback = HistoryCallback(approx_order = approximation_order)
# analysis_interval = 100
# analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
save_solution = SolutionSavingCallback(dt = 0.01,
                                       prefix = savename)
# save_solution = SolutionSavingCallback(interval = 10,
#                                        prefix = savename)
callbacks = CallbackSet(summary_callback, alive_callback, save_solution)
time_int_tol = 1e-3
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                     variables = (pressure, Trixi.density))
# Solve
sol = solve(ode, SSPRK43(stage_limiter! = stage_limiter!); abstol = time_int_tol,
            reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
# sol = solve(ode, SSPRK54(stage_limiter! = stage_limiter!); dt = 0.000001,
#             abstol = time_int_tol, reltol = time_int_tol,
#             ode_default_options()..., callback = callbacks)
summary_callback()
