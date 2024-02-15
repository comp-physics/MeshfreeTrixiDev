using Revise
using StaticArrays

struct BoundaryData{Ti<:Integer,Tv<:SVector{N,T} where {N,T<:Number}}
    idx::Vector{Ti}       # Indices of boundary points
    normals::Vector{Tv}   # Normals at boundary points
end

struct PointCloudDomain{Dim,Tv,Ti}
    points::Vector{Tv}  # Point coordinates
    neighbors::Vector{Vector{Ti}}  # Neighbors for each point
    boundary_tags::Dict{Symbol,BoundaryData{Ti,Tv}}  # Adjusted to use BoundaryData
end

function PointCloudDomain(points::Vector{Tv}, neighbors::Vector{Vector{Ti}},
    boundary_tags::Dict{Symbol,BoundaryData{Ti,Tv}}) where {N,Tv<:SVector{N,Float64},Ti}
    return PointCloudDomain{N,Tv,Ti}(points, neighbors, boundary_tags)
end




