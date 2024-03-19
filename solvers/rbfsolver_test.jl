using Revise
using Trixi
using ConstructionBase
using MuladdMacro
using Trixi: @threaded
using Trixi: @trixi_timeit
using Trixi: summary_header, summary_line, summary_footer, increment_indent
using Trixi: True, False
using NearestNeighbors
using LinearAlgebra
using SparseArrays
using StructArrays

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
