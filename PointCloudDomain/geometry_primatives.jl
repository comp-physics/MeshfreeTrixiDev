# NodesandModes for PointCloudDomain
# each type of element shape - used for dispatch only (basis specifically)
abstract type AbstractElemShape{NDIMS} end
struct Point1D <: AbstractElemShape{1} end
struct Point2D <: AbstractElemShape{2} end
struct Point3D <: AbstractElemShape{3} end

# StartUpDG for PointCloudDomain. Uses base types from NodesAndModes
"""
    struct RefElemData

RefElemData: contains info (point coords, neighbors, order of accuracy)
for a high order RBF basis on a given reference element. 

Example:
```julia
N = 3
rd = RefElemData(Tri(), N)
(; r, s ) = rd
```
"""
### RefElemData called by basis
# Rework for PointCloudDomain
struct RefElemData{Dim, ElemShape <: AbstractElemShape{Dim}, ApproximationType,
                   NT, NV, PV}
    element_type::ElemShape
    approximation_type::ApproximationType # Polynomial / SBP{...}

    N::NT               # polynomial degree of accuracy
    nv::NV               # list of vertices defining neighbors
    pv::PV               # point coordinates
end

# need this to use @set outside of StartUpDG
function ConstructionBase.setproperties(rd::RefElemData, patch::NamedTuple)
    fields = (haskey(patch, symbol) ? getproperty(patch, symbol) : getproperty(rd, symbol) for symbol in fieldnames(typeof(rd)))
    return RefElemData(fields...)
end

function ConstructionBase.getproperties(rd::RefElemData)
    (; element_type = rd.element_type, approximation_type = rd.approximation_type, N = rd.N,
     nv = rd.nv,
     pv = rd.pv)
end

# Updated _propertynames function to reflect the new fields in RefElemData
_propertynames(::Type{RefElemData}, private::Bool = false) = (:nv, :pv)

function Base.propertynames(x::RefElemData, private::Bool = false)
    return (fieldnames(typeof(x))..., _propertynames(typeof(x))...)
end

# convenience unpacking routines
# Not necessary for PointCloudDomain
# function Base.getproperty(x::RefElemData{Dim, ElementType, ApproxType},
#                           s::Symbol) where {Dim, ElementType, ApproxType}
#     return getfield(x, s)
# end

"""
    function RefElemData(elem; N, kwargs...)
    function RefElemData(elem, approx_type; N, kwargs...)

Keyword argument constructor for RefElemData (to "label" `N` via `rd = RefElemData(Line(), N=3)`)
"""
RefElemData(elem; N, kwargs...) = RefElemData(elem, N; kwargs...)
RefElemData(elem, approx_type; N, kwargs...) = RefElemData(elem, approx_type, N; kwargs...)

# default to Polynomial-type RefElemData
RefElemData(elem, N::Int; kwargs...) = RefElemData(elem, RBF(), N; kwargs...)

@inline Base.ndims(::Point1D) = 1
@inline Base.ndims(::Point2D) = 2
@inline Base.ndims(::Point3D) = 3

# ====================================================
#          RefElemData approximation types
# ====================================================

"""
    RBF{T}

Represents RBF approximation types (as opposed to generic polynomials). 
By default, `RBF()` constructs a `RBF{DefaultRBFType}`.
Specifying a type parameters allows for dispatch on additional structure within an
RBF approximation (e.g., polyharmonic spline, collocation, etc). 
"""
struct RBF{T}
    data::T
end

struct DefaultRBFType end
RBF() = RBF{DefaultRBFType}(DefaultRBFType())

# Polynomial{Gauss} type indicates (N+1)-point Gauss quadrature on tensor product elements
struct PolyharmonicSpline end
RBF{PolyharmonicSpline}() = RBF(PolyharmonicSpline())

# ====================================
#              Printing 
# ====================================

function Base.show(io::IO, ::MIME"text/plain", rd::RefElemData)
    @nospecialize rd
    print(io,
          "RefElemData for a degree $(rd.N) $(_short_typeof(rd.approximation_type)) " *
          "approximation on a $(_short_typeof(rd.element_type)) element.")
end

function Base.show(io::IO, rd::RefElemData)
    @nospecialize basis # reduce precompilation time
    print(io,
          "RefElemData{N=$(rd.N), $(_short_typeof(rd.approximation_type)), $(_short_typeof(rd.element_type))}.")
end

_short_typeof(x) = typeof(x)

_short_typeof(approx_type::RBF{<:DefaultRBFType}) = "RBF"
_short_typeof(approx_type::RBF{<:PolyharmonicSpline}) = "RBF{PolyharmonicSpline}"
# function _short_typeof(approx_type::Polynomial{<:TensorProductQuadrature})
#     "Polynomial{TensorProductQuadrature}"
# end

"""
    RefElemData(elem::Line, N;
                quad_rule_vol = quad_nodes(elem, N+1))
    RefElemData(elem::Union{Tri, Quad}, N;
                 quad_rule_vol = quad_nodes(elem, N),
                 quad_rule_face = gauss_quad(0, 0, N))
    RefElemData(elem::Union{Hex, Tet}, N;
                 quad_rule_vol = quad_nodes(elem, N),
                 quad_rule_face = quad_nodes(Quad(), N))
    RefElemData(elem; N, kwargs...) # version with keyword args

Constructor for `RefElemData` for different element types.
"""