using Revise
using Trixi
using ConstructionBase
using MuladdMacro
using Base.Threads

includet("../header.jl")

basis = RefPointData(Point1D(), RBF(DefaultRBFType(5)), 5)
basis = RefPointData(Point1D(), RBF(), 5)

basis = RefPointData(Point1D(), RBF(PolyharmonicSpline(5)), 3)
rbf = RBFSolver(basis, nothing, nothing, nothing)