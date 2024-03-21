using Revise

includet("../header.jl")

# Base Methods
basis = RefPointData(Point1D(), RBF(DefaultRBFType(5)), 5)
basis = RefPointData(Point2D(), RBF(), 5)
basis = RefPointData(Point1D(), RBF(PolyharmonicSpline(5)), 3)
solver = RBFSolver(basis, RBFFDEngine())

# Specialized Methods
basis = PointCloudBasis(Point2D(), 3; approximation_type = RBF(PolyharmonicSpline(5)))
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
initial_condition = initial_condition_constant
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => BoundaryConditionNeumann(initial_condition),
                       :rest => boundary_condition_slip_wall)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

# Working through cache and operator details
includet("../utilities/helper.jl")
show_ft(semi.cache)
u = deepcopy(ode.u0)
du = deepcopy(ode.u0)
n = length(u)
A = sprand(n, n, 0.01)
apply_to_each_field(mul_by!(A), du, u)

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