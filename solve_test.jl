using Revise
using MeshfreeTrixi
using OrdinaryDiffEq

# includet("../header.jl")

# Base Methods
approximation_order = 1
rbf_order = 3
# Specialized Methods
basis = PointCloudBasis(Point2D(), approximation_order;
                        approximation_type = RBF(PolyharmonicSpline(rbf_order)))
solver = PointCloudSolver(basis)

casename = "./medusa_point_clouds/cyl_0_0125"
boundary_names = Dict(:inlet => 1, :outlet => 2, :bottom => 3, :top => 4, :cyl => 5)
domain = PointCloudDomain(solver, casename, boundary_names)

# Instantiate Semidiscretization
function basic_limiter!(u_ode, integrator,
                        semi::Trixi.AbstractSemidiscretization,
                        t)
    @unpack mesh, solver, cache, equations = semi
    for e in eachelement(mesh, solver, cache)
        rho, rho_v1, rho_v2, rho_e = u_ode[e]
        if rho < 0.0
            rho = eps()
        end
        if rho_e < 0.0
            rho_e = eps()
        end
        # p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
        u_ode[e] = SVector(rho, rho_v1, rho_v2, rho_e)
    end
end
equations = CompressibleEulerEquations2D(1.4)
function initial_condition_cyl(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.4
    rho_v1 = 4.1
    rho_v2 = 0.0
    rho_e = 8.8 * 1.4
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function initial_condition_gradient(x, t, equations::CompressibleEulerEquations2D)
    rho_s = 1.4
    rho_v1_s = 4.1
    rho_v2_s = 0.0
    rho_e_s = 8.8
    rho = rho_s + 0.1 * x[1] + 0.1 * x[2]
    rho_v1 = rho_v1_s + 0.1 * x[1] + 0.1 * x[2]
    rho_v2 = rho_v2_s + 0.1 * x[1] + 0.1 * x[2]
    rho_e = rho_e_s + 0.1 * x[1] + 0.1 * x[2]
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
initial_condition = initial_condition_cyl
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => BoundaryConditionDoNothing(),
                       :top => boundary_condition_slip_wall,
                       :bottom => boundary_condition_slip_wall,
                       :cyl => boundary_condition_slip_wall)

# Test upwind viscosity
source_rv = SourceUpwindViscosityTominec(solver, equations, domain)
source_hv2 = SourceHyperviscosityTominec(solver, equations, domain; c = 1.0)
sources = SourceTerms(hv = source_hv2, rv = source_rv)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

# Try sim
# summary_callback = SummaryCallback()
summary_callback = InfoCallback()
alive_callback = AliveCallback(alive_interval = 10)
# analysis_interval = 100
# analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
save_solution = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
# save_solution = SaveSolutionCallback(interval = 100,
#                                      save_initial_solution = true,
#                                      save_final_solution = true,
#                                      solution_variables = cons2prim)
callbacks = CallbackSet(summary_callback, alive_callback)
time_int_tol = 1e-3
sol = solve(ode, SSPRK43(); abstol = time_int_tol,
            reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
sol = solve(ode, SSPRK54(); dt = 0.00001,
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
summary_callback()
# Plotting
using GLMakie
time = 0.0
rho = [sol[end][x][1] for x in 1:length(sol.u[end])]
# rho = [sol(time)[x][1] for x in 1:length(sol.u[end])]
mx = [sol[end][x][2] for x in 1:length(sol.u[end])]
my = [sol[end][x][3] for x in 1:length(sol.u[end])]
rho_e = [sol[end][x][4] for x in 1:length(sol.u[end])]
any(isnan.(rho))
rho[isnan.(rho)] .= 0.0
scatter(domain.pd.points, color = rho, axis = (aspect = DataAspect(),))
scatter(domain.pd.points, color = mx, axis = (aspect = DataAspect(),))
scatter(domain.pd.points, color = my, axis = (aspect = DataAspect(),))
scatter(domain.pd.points, color = rho_e, axis = (aspect = DataAspect(),))
domain.pd.points[semi.mesh.boundary_tags[:cyl].idx]
semi.mesh.boundary_tags[:cyl].idx
semi.mesh.boundary_tags[:cyl].normals
mx[semi.mesh.boundary_tags[:cyl].idx]

# Test single step
u0 = ode.u0
u = deepcopy(u0)
du = deepcopy(u0)
# Function
t = 0.0
MeshfreeTrixi.rhs!(du, u, semi, t)
# Separate functions
@unpack cache, boundary_conditions, source_terms = semi
t = 0.0
MeshfreeTrixi.reset_du!(du, solver, cache)
MeshfreeTrixi.calc_boundary_flux!(du, u, cache, t, boundary_conditions, domain,
                                  MeshfreeTrixi.have_nonconservative_terms(equations),
                                  equations, solver)
MeshfreeTrixi.calc_fluxes!(du, u, domain,
                           MeshfreeTrixi.have_nonconservative_terms(equations), equations,
                           solver.engine, solver, cache)
MeshfreeTrixi.calc_sources!(du, u, t, source_terms, domain, equations, solver, cache)
MeshfreeTrixi.calc_boundary_flux!(du, u, cache, t, boundary_conditions, domain,
                                  MeshfreeTrixi.have_nonconservative_terms(equations),
                                  equations, solver)
u[semi.mesh.boundary_tags[:cyl].idx]
du[semi.mesh.boundary_tags[:cyl].idx]
# Plotting
using GLMakie
time = 0.0
rho = [du[x][1] for x in 1:length(du)]
# rho = [sol(time)[x][1] for x in 1:length(sol.u[end])]
mx = [du[x][2] for x in 1:length(du)]
my = [du[x][3] for x in 1:length(du)]
rho_e = [du[x][4] for x in 1:length(du)]
any(isnan.(rho))
rho[isnan.(rho)] .= 0.0
scatter(domain.pd.points, color = rho, axis = (aspect = DataAspect(),))
scatter(domain.pd.points, color = mx, axis = (aspect = DataAspect(),))
scatter(domain.pd.points, color = my, axis = (aspect = DataAspect(),))
scatter(domain.pd.points, color = rho_e, axis = (aspect = DataAspect(),))

# Plotting areas of negative pressure
idx = 8361
scatter(domain.pd.points, axis = (aspect = DataAspect(),))
scatter!(domain.pd.points[idx], color = :red)
