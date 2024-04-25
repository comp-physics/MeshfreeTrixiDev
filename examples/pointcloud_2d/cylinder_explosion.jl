using Revise
using MeshfreeTrixi
using OrdinaryDiffEq

# includet("../header.jl")

# Base Methods
approximation_order = 3
rbf_order = 3
# Specialized Methods
basis = PointCloudBasis(Point2D(), approximation_order;
                        approximation_type = RBF(PolyharmonicSpline(rbf_order)))
solver = PointCloudSolver(basis)

dir = "./medusa_point_clouds"
casename = "cyl_explosion_0_00625"
domain_name = joinpath(dir, casename)
savename = casename * "_order_$approximation_order"
boundary_names = Dict(:outer => 1, :inner => 2)
# boundary_names = Dict(:all => 1)
domain = PointCloudDomain(solver, domain_name, boundary_names)

# scatter(domain.pd.points, axis = (aspect = DataAspect(),))
# scatter!(domain.pd.points[domain.boundary_tags[:inlet].idx], color = :black)
# for tag in keys(domain.boundary_tags)
#     idx = domain.boundary_tags[tag].idx
#     scatter!(domain.pd.points[idx], label = string(tag))
# end

equations = CompressibleEulerEquations2D(1.4)
function initial_condition_cyl(x, t, equations::CompressibleEulerEquations2D)
    r = sqrt(sum(x .^ 2))
    if r < sqrt(0.4)
        rho = 1
        vx = 0
        vy = 0
        p = 1
        return prim2cons(SVector(rho, vx, vy, p), equations)
    else
        rho = 0.125
        vx = 0
        vy = 0
        p = 0.1
        return prim2cons(SVector(rho, vx, vy, p), equations)
    end
end
initial_condition = initial_condition_cyl
boundary_conditions = (; :outer => boundary_condition_slip_wall,
                       :inner => boundary_condition_slip_wall)

# Test upwind viscosity
source_rv = SourceResidualViscosityTominec(solver, equations, domain; c_rv = 5.0,
                                           c_uw = 1.0, polydeg = approximation_order)
# source_rv = SourceUpwindViscosityTominec(solver, equations, domain; c_uw = 1.0)
source_hv = SourceHyperviscosityTominec(solver, equations, domain;
                                        c = domain.pd.dx_min^(-2 + 0.5))
# source_hv = SourceHyperviscosityFlyer(solver, equations, domain;
#                                       k = 2,
#                                       c = domain.pd.dx_min^(-2 + 0.0))
sources = SourceTerms(hv = source_hv, rv = source_rv)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)
tspan = (0.0, 4.0)
ode = semidiscretize(semi, tspan)

# Try sim
# summary_callback = SummaryCallback()
summary_callback = InfoCallback()
alive_callback = AliveCallback(alive_interval = 100)
analysis_interval = 1000
performance_callback = PerformanceCallback(semi, interval = analysis_interval,
                                           uEltype = real(solver))
history_callback = HistoryCallback(approx_order = approximation_order)
# analysis_interval = 100
# analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
save_solution = SolutionSavingCallback(dt = 0.1,
                                       prefix = savename)
# save_solution = SolutionSavingCallback(interval = 10,
#                                        prefix = savename)
callbacks = CallbackSet(summary_callback, alive_callback, performance_callback,
                        history_callback, save_solution)
time_int_tol = 1e-3
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                     variables = (pressure, Trixi.density))
# Solve
sol = solve(ode, SSPRK43(stage_limiter! = stage_limiter!); abstol = time_int_tol,
            reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
# sol = solve(ode, SSPRK22(stage_limiter! = stage_limiter!); dt = 0.00001,
#             abstol = time_int_tol, reltol = time_int_tol,
#             ode_default_options()..., callback = callbacks)
summary_callback()
