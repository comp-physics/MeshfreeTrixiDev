using Revise
using MeshfreeTrixi
using OrdinaryDiffEq

# includet("../header.jl")

# Base Methods
approximation_order = 5
rbf_order = 3
# basis = RefPointData(Point1D(), RBF(DefaultRBFType(5)), approximation_order)
basis = RefPointData(Point2D(), RBF(), approximation_order)
basis = RefPointData(Point1D(), RBF(PolyharmonicSpline(rbf_order)), approximation_order)
solver = RBFSolver(basis, RBFFDEngine())

# Specialized Methods
basis = PointCloudBasis(Point2D(), approximation_order;
                        approximation_type = RBF(PolyharmonicSpline(rbf_order)))
solver = PointCloudSolver(basis)

casename = "./medusa_point_clouds/cyl"
boundary_names = Dict(:inlet => 1, :outlet => 2, :bottom => 3, :top => 4, :cyl => 5)
domain = PointCloudDomain(solver, casename, boundary_names)

# # Plot
# using GLMakie
# scatter(Tuple.(domain.pd.points), color = :black, markersize = 10.0, marker = :circle,
#         axis = (aspect = DataAspect(),))
# # # Plot boundaries
# key = :cyl
# boundary = domain.pd.points[domain.boundary_tags[key].idx]
# normals = domain.boundary_tags[key].normals
# boundary_x = getindex.(boundary, 1)
# boundary_y = getindex.(boundary, 2)
# normals_dx = getindex.(normals, 1)
# normals_dy = getindex.(normals, 2)
# scatter!(boundary_x, boundary_y, markersize = 10.0)
# quiver!(boundary_x, boundary_y, normals_dx, normals_dy, lengthscale = 0.05)

# Instantiate source terms here?
# We need access to solver, basis, and domain 
# since source_hyperviscosity(solver, equations, domain) 
# and source_tominec_rv(solver, equations, domain)
# require basis info for bases and domain info for 
# create the actual operators. Also eqn info for 
# number of eqn perhaps?

# Also need a specific callback to update the time history
# history_callback = HistoryCallback()

# Instantiate Semidiscretization
equations = CompressibleEulerEquations2D(1.4)
function initial_condition_cyl(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.4
    rho_v1 = 4.1
    rho_v2 = 0.0
    rho_e = 8.8 #* 1.4
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
initial_condition = initial_condition_cyl
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => Trixi.BoundaryConditionDoNothing(),
                       :top => boundary_condition_slip_wall,
                       :bottom => boundary_condition_slip_wall,
                       :cyl => boundary_condition_slip_wall)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

# Working through cache and operator details
# includet("../utilities/helper.jl")
# show_ft(semi.cache)
# u = deepcopy(ode.u0)
# du = deepcopy(ode.u0)
# n = length(u)
# A = sprand(n, n, 0.01)
# apply_to_each_field(mul_by!(A), du, u)

# Test operator instantiation
# A = rbf_block(basis.f.rbf, basis; N = 3)
rbf_func = concrete_rbf_flux_basis(basis.f.rbf, basis)
poly_func = concrete_poly_flux_basis(basis.f.poly, basis)
rbf_differentiation_matrices = compute_flux_operator(solver, domain)

# Test source_terms
source_hv = SourceHyperviscosityFlyer(solver, equations, domain; k = 2, c = 1.0)
source_hv2 = SourceHyperviscosityTominec(solver, equations, domain; c = 1.0)
sources = (; source_hv, source_hv2)
sources = SourceTerms(hv = source_hv, hv2 = source_hv2)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

# Test history callback
history_callback = HistoryCallback(approx_order = approximation_order)
source_rv = SourceResidualViscosityTominec(solver, equations, domain; c = 1.0,
                                           polydeg = approximation_order)
sources = SourceTerms(hv = source_hv2, rv = source_rv)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

# Try sim
# summary_callback = SummaryCallback()
summary_callback = InfoCallback()
alive_callback = AliveCallback(alive_interval = 10)
# analysis_interval = 100
# analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, history_callback)
time_int_tol = 1e-8
sol = solve(ode, SSPRK43(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)

# Test upwind viscosity
source_rv = SourceUpwindViscosityTominec(solver, equations, domain)
source_hv2 = SourceHyperviscosityTominec(solver, equations, domain; c = 1.0)
sources = SourceTerms(hv = source_hv2, rv = source_rv)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

# Try sim
# summary_callback = SummaryCallback()
summary_callback = InfoCallback()
alive_callback = AliveCallback(alive_interval = 10)
# analysis_interval = 100
# analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback)
time_int_tol = 1e-8
sol = solve(ode, SSPRK43(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)

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

# Plotting areas of negative pressure
scatter(domain.pd.points, axis = (aspect = DataAspect(),))
scatter!(domain.pd.points[idx], color = :red)
