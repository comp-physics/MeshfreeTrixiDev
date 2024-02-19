using Revise
using Trixi
using ConstructionBase

includet("geometry_primatives.jl")

basis = RefElemData(Point1D(), RBF(DefaultRBFType()), 3)