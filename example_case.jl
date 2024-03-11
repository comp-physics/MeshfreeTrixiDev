using Revise
using Trixi
using ConstructionBase
using MuladdMacro
using Trixi: @threaded

includet("basic_types.jl")
includet("PointCloudDomain/geometry_primatives.jl")
includet("PointCloudDomain/PointCloudDomain.jl")
includet("solvers/rbfsolver.jl")
includet("solvers/pointcloudsolver/types.jl")
includet("solvers/pointcloudsolver/rbfsolver.jl")

"""
Basic example case of our RBF-FD point cloud methods in the structure of Trixi. Formulated using DGMulti as a template. Domain object replaces mesh object. Volume Integral is replaced with RBF Engine and employs underlying numerics of RBF-FD.  

## References

- Flyer (2016)
  Enhancing finite differences with radial basis functions: 
  Experiments on the Navier-Stokes equations
  [doi: 10.1016/j.jcp.2016.02.078](https://doi.org/10.1016/j.jcp.2016.02.078)
- Tominec, Murtazo (2022)
  Residual Viscosity Stabilized RBF-FD Methods for Solving
  Nonlinear Conservation Laws
  [doi: 10.1007/s10915-022-02055-8](https://doi.org/10.1007/s10915-022-02055-8)

Point Cloud based simulation requires the use of a stabilizer such as the Tominec AV operator in conjunction with Flyer HV operator. 
"""

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

# surface_flux = flux_lax_friedrichs
# volume_flux = flux_ranocha

polydeg = 3
# basis = DGMultiBasis(Quad(), polydeg, approximation_type = GaussSBP())
basis = RefPointData(Point1D(), RBF(PolyharmonicSpline(5)), 3) # Make specialized with PointCloudBasis

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
engine = RBFFDEngine(indicator_sc)

solver = PointCloudSolver(basis,
                          engine = engine)

# mesh = PointCloudDomain(solver, cells_per_dimension, periodicity = false)
mesh = MedusaPointCloud{2}(filename) # Specialized PointCloudDomain

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)