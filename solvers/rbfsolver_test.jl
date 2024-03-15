using Revise
using Trixi
using ConstructionBase
using MuladdMacro
using Trixi: @threaded
using Trixi: summary_header, summary_line, summary_footer, increment_indent

includet("../header.jl")

basis = RefPointData(Point1D(), RBF(DefaultRBFType(5)), 5)
basis = RefPointData(Point1D(), RBF(), 5)

basis = RefPointData(Point1D(), RBF(PolyharmonicSpline(5)), 3)
rbf = RBFSolver(basis, RBFFDEngine())

PointCloudBasis(Point1D(), 3; approximation_type = RBF(PolyharmonicSpline(5)))