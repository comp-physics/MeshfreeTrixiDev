using Revise
using Trixi
using ConstructionBase
using MuladdMacro
using Trixi: @threaded
using Trixi: @trixi_timeit
using Trixi: summary_header, summary_line, summary_footer, increment_indent
using NearestNeighbors
using GLMakie

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

# Plot
scatter(Tuple.(domain.pd.points), color = :black, markersize = 10.0, marker = :circle,
        axis = (aspect = DataAspect(),))
# # Plot boundaries
# boundary = Tuple.(domain.pd.points[domain.boundary_tags[:cyl].idx])
# scatter!(boundary, markersize = 15.0, marker = :circle)
# for i in eachindex(boundary_idxs)
#     boundary = Tuple.(positions[boundary_idxs[i]])
#     scatter!(boundary, markersize = 15.0, marker = :circle)
# end

# Extract boundary points and normals
key = :cyl
boundary = domain.pd.points[domain.boundary_tags[key].idx]
normals = domain.boundary_tags[key].normals
# Decompose boundary points and normals into their components
boundary_x = getindex.(boundary, 1)
boundary_y = getindex.(boundary, 2)
normals_dx = getindex.(normals, 1)
normals_dy = getindex.(normals, 2)
# Plot the boundary points
scatter!(boundary_x, boundary_y, markersize = 10.0)
# Add the quiver plot for normals
quiver!(boundary_x, boundary_y, normals_dx, normals_dy, lengthscale = 0.05)
# Display the figure