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

includet("../header.jl")

# Base Methods
basis = RefPointData(Point1D(), RBF(DefaultRBFType(5)), 5)
basis = RefPointData(Point2D(), RBF(), 5)
basis = RefPointData(Point1D(), RBF(PolyharmonicSpline(5)), 3)
rbf = RBFSolver(basis, RBFFDEngine())

# Specialized Methods
basis = PointCloudBasis(Point2D(), 3; approximation_type = RBF(PolyharmonicSpline(5)))
rbf = PointCloudSolver(basis)

casename = "./medusa_point_clouds/cyl"
boundary_names = Dict(:inlet => 1, :outlet => 2, :bottom => 3, :top => 4, :cyl => 5)
domain = PointCloudDomain(rbf, casename, boundary_names)

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
initial_condition = initial_condition_weak_blast_wave
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => BoundaryConditionNeumann(initial_condition),
                       :rest => boundary_condition_slip_wall)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, rbf;
                                    boundary_conditions = boundary_conditions)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)