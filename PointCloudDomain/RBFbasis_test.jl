using Revise
using Trixi
using ConstructionBase

includet("geometry_primatives.jl")

basis = RefElemData(Point1D(), RBF(DefaultRBFType(5)), 5)
basis = RefElemData(Point1D(), RBF(), 5)

basis = RefElemData(Point1D(), RBF(PolyharmonicSpline(5)), 3)