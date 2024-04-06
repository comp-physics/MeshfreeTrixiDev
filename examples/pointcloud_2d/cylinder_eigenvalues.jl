using Revise
using MeshfreeTrixi
using OrdinaryDiffEq
using Arpack
using SparseArrays
using LinearAlgebra
using GLMakie

# includet("../header.jl")

# Base Methods
approximation_order = 3
rbf_order = 3
# Specialized Methods
basis = PointCloudBasis(Point2D(), approximation_order;
                        approximation_type = RBF(PolyharmonicSpline(rbf_order)))
solver = PointCloudSolver(basis)

dir = "./medusa_point_clouds"
casename = "cyl_0_0125"
domain_name = joinpath(dir, casename)
savename = casename * "_order_$approximation_order"
boundary_names = Dict(:inlet => 1, :outlet => 2, :bottom => 3, :top => 4, :cyl => 5)
domain = PointCloudDomain(solver, domain_name, boundary_names)

equations = CompressibleEulerEquations2D(1.4)
function initial_condition_cyl(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.4
    rho_v1 = 4.1
    rho_v2 = 0.0
    rho_e = 8.8
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
initial_condition = initial_condition_cyl
boundary_conditions = (; :inlet => BoundaryConditionDirichlet(initial_condition),
                       :outlet => BoundaryConditionDoNothing(),
                       :top => boundary_condition_slip_wall,
                       :bottom => boundary_condition_slip_wall,
                       :cyl => boundary_condition_slip_wall)

# Test upwind viscosity
source_rv = SourceResidualViscosityTominec(solver, equations, domain; c_rv = 0.1,
                                           c_uw = 1.0, polydeg = approximation_order + 1)
# source_rv = SourceUpwindViscosityTominec(solver, equations, domain; c_uw = 1.0)
source_hv2 = SourceHyperviscosityTominec(solver, equations, domain;
                                         c = domain.pd.dx_min^(-2 - 2.0))
sources = SourceTerms(hv = source_hv2, rv = source_rv)
semi = SemidiscretizationHyperbolic(domain, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = sources)

# Extract Eigenvalues
Dx = semi.cache.rbf_differentiation_matrices[1]
Dy = semi.cache.rbf_differentiation_matrices[2]
eigenvalues, phi = eigs(Dx, nev = 1000)
scatter(real.(eigenvalues), imag.(eigenvalues))