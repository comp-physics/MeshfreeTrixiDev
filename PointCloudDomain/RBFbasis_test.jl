using Revise
using Trixi
using ConstructionBase

includet("geometry_primatives.jl")

basis = RefElemData(Point1D(), RBF(DefaultRBFType()), 3)
basis = RefElemData(Point1D(), RBF(), 3)

basis = RefElemData(Point1D(), RBF(PolyharmonicSpline(5)), 3)