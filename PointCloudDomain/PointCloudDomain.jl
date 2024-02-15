using Revise
using StaticArrays

struct PointCloudDomain{Dim,Tv,Ti}
    points::Vector{Tv}  # Point coordinates
    neighbors::Vector{Vector{Ti}}  # Neighbors for each point
    boundary_tags::Dict{Symbol,Tuple{Vector{Ti},Vector{Tv}}}  # Boundary tags with indices and normals of boundary points
end

# Outer constructor
# function PointCloudDomain(points::Vector{Tv}, neighbors::Vector{Vector{Ti}},
#     boundary_tags::Dict{Symbol,Tuple{Vector{Ti},Vector{Tv}}}) where {Tv<:SVector{N,Float64},Ti}
#     return PointCloudDomain{N,Tv,Ti}(points, neighbors, boundary_tags)
# end

PointCloudDomain(points::Vector{Tv}, neighbors::Vector{Vector{Ti}},
    boundary_tags::Dict{Symbol,Tuple{Vector{Ti},Vector{Tv}}}) where {N,Tv<:SVector{N,Float64},Ti} =
    PointCloudDomain{N,Tv,Ti}(points, neighbors, boundary_tags)




