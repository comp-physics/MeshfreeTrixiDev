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

_propertynames(::Type{RefElemData}, private::Bool = false) = (:num_faces, :Np, :Nq, :Nfq)
function Base.propertynames(x::RefElemData{1}, private::Bool = false)
    return (fieldnames(RefElemData)..., _propertynames(RefElemData)...,
            :r, :rq, :rf, :rp, :nrJ, :Dr)
end
function Base.propertynames(x::RefElemData{2}, private::Bool = false)
    return (fieldnames(RefElemData)..., _propertynames(RefElemData)...,
            :r, :s, :rq, :sq, :rf, :sf, :rp, :sp, :nrJ, :nsJ, :Dr, :Ds)
end
function Base.propertynames(x::RefElemData{3}, private::Bool = false)
    return (fieldnames(RefElemData)..., _propertynames(RefElemData)...,
            :r, :s, :t, :rq, :sq, :tq, :rf, :sf, :tf,
            :rp, :sp, :tp, :nrJ, :nsJ, :ntJ, :Dr, :Ds, :Dt)
end

# convenience unpacking routines
function Base.getproperty(x::RefElemData{Dim, ElementType, ApproxType},
                          s::Symbol) where {Dim, ElementType, ApproxType}
    if s == :r
        return getfield(x, :rst)[1]
    elseif s == :s
        return getfield(x, :rst)[2]
    elseif s == :t
        return getfield(x, :rst)[3]

    elseif s == :rq
        return getfield(x, :rstq)[1]
    elseif s == :sq
        return getfield(x, :rstq)[2]
    elseif s == :tq
        return getfield(x, :rstq)[3]

    elseif s == :rf
        return getfield(x, :rstf)[1]
    elseif s == :sf
        return getfield(x, :rstf)[2]
    elseif s == :tf
        return getfield(x, :rstf)[3]

    elseif s == :rp
        return getfield(x, :rstp)[1]
    elseif s == :sp
        return getfield(x, :rstp)[2]
    elseif s == :tp
        return getfield(x, :rstp)[3]

    elseif s == :nrJ
        return getfield(x, :nrstJ)[1]
    elseif s == :nsJ
        return getfield(x, :nrstJ)[2]
    elseif s == :ntJ
        return getfield(x, :nrstJ)[3]

    elseif s == :Dr
        return getfield(x, :Drst)[1]
    elseif s == :Ds
        return getfield(x, :Drst)[2]
    elseif s == :Dt
        return getfield(x, :Drst)[3]

    elseif s == :Nfaces || s == :num_faces
        return num_faces(getfield(x, :element_type))
    elseif s == :Np
        return length(getfield(x, :rst)[1])
    elseif s == :Nq
        return length(getfield(x, :rstq)[1])
    elseif s == :Nfq
        return length(getfield(x, :rstf)[1])
    else
        return getfield(x, s)
    end
end

"""
    function RefElemData(elem; N, kwargs...)
    function RefElemData(elem, approx_type; N, kwargs...)

Keyword argument constructor for RefElemData (to "label" `N` via `rd = RefElemData(Line(), N=3)`)
"""
RefElemData(elem; N, kwargs...) = RefElemData(elem, N; kwargs...)
RefElemData(elem, approx_type; N, kwargs...) = RefElemData(elem, approx_type, N; kwargs...)

# default to Polynomial-type RefElemData
RefElemData(elem, N::Int; kwargs...) = RefElemData(elem, Polynomial(), N; kwargs...)

@inline Base.ndims(::Point1D) = 1
@inline Base.ndims(::Point2D) = 2
@inline Base.ndims(::Point3D) = 3

# @inline num_vertices(::Tri) = 3
# @inline num_vertices(::Union{Quad, Tet}) = 4
# @inline num_vertices(::Hex) = 8
# @inline num_vertices(::Wedge) = 6
# @inline num_vertices(::Pyr) = 5

# @inline num_faces(::Line) = 2
# @inline num_faces(::Tri) = 3
# @inline num_faces(::Union{Quad, Tet}) = 4
# @inline num_faces(::Union{Wedge, Pyr}) = 5
# @inline num_faces(::Hex) = 6

# @inline face_type(::Union{Tri, Quad}) = Line()
# @inline face_type(::Hex) = Quad()
# @inline face_type(::Tet) = Tri()

# generic fallback 
# @inline face_type(elem::AbstractElemShape, id) = face_type(elem)

# Wedges have different types of faces depending on the face. 
# We define the first three faces to be quadrilaterals and the 
# last two faces are triangles.
# @inline face_type(::Wedge, id) = (id <= 3) ? Quad() : Tri()

# Pyramids have different types of faces depending on the face. 
# We define the first four faces to be triangles and the 
# last face to be a quadrilateral. 
# @inline face_type(::Pyr, id) = (id <= 4) ? Tri() : Quad()

# ====================================================
#          RefElemData approximation types
# ====================================================

"""
    Polynomial{T}

Represents polynomial approximation types (as opposed to finite differences). 
By default, `Polynomial()` constructs a `Polynomial{StartUpDG.DefaultPolynomialType}`.
Specifying a type parameters allows for dispatch on additional structure within a
polynomial approximation (e.g., collocation, tensor product quadrature, etc). 
"""
struct Polynomial{T}
    data::T
end

struct DefaultPolynomialType end
Polynomial() = Polynomial{DefaultPolynomialType}(DefaultPolynomialType())

"""
    TensorProductQuadrature{T}

A type parameter to `Polynomial` indicating that 
"""
struct TensorProductQuadrature{T}
    quad_rule_1D::T  # 1D quadrature nodes and weights (rq, wq)
end

TensorProductQuadrature(r1D, w1D) = TensorProductQuadrature((r1D, w1D))

# Polynomial{Gauss} type indicates (N+1)-point Gauss quadrature on tensor product elements
struct Gauss end
Polynomial{Gauss}() = Polynomial(Gauss())

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

_short_typeof(approx_type::Wedge) = "Wedge"
_short_typeof(approx_type::Pyr) = "Pyr"

_short_typeof(approx_type::Polynomial{<:DefaultPolynomialType}) = "Polynomial"
_short_typeof(approx_type::Polynomial{<:Gauss}) = "Polynomial{Gauss}"
function _short_typeof(approx_type::Polynomial{<:TensorProductQuadrature})
    "Polynomial{TensorProductQuadrature}"
end