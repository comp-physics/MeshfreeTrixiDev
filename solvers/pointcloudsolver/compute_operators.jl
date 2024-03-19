function concrete_rbf_flux_basis(rbf, basis::RefPointData{NDIMS}) where {NDIMS}
    if NDIMS == 1
        @variables x
        Dx = Differential(x)
        rbf_x = simplify(expand_derivatives(Dx(rbf)))
        rbf_expr = build_function(rbf, [x]; expression = Val{false})
        rbf_x_expr = build_function(rbf_x, [x, y]; expression = Val{false})
        return (; rbf_expr, rbf_x_expr)
    elseif NDIMS == 2
        @variables x y
        Dx = Differential(x)
        Dy = Differential(y)
        rbf_x = simplify(expand_derivatives(Dx(rbf)))
        rbf_y = simplify(expand_derivatives(Dy(rbf)))
        rbf_expr = build_function(rbf, [x, y]; expression = Val{false})
        rbf_x_expr = build_function(rbf_x, [x, y]; expression = Val{false})
        rbf_y_expr = build_function(rbf_y, [x, y]; expression = Val{false})
        return (; rbf_expr, rbf_x_expr, rbf_y_expr)
    elseif NDIMS == 3
        @variables x y z
        Dx = Differential(x)
        Dy = Differential(y)
        Dz = Differential(z)
        rbf_x = simplify(expand_derivatives(Dx(rbf)))
        rbf_y = simplify(expand_derivatives(Dy(rbf)))
        rbf_z = simplify(expand_derivatives(Dz(rbf)))
        rbf_expr = build_function(rbf, [x, y, z]; expression = Val{false})
        rbf_x_expr = build_function(rbf_x, [x, y]; expression = Val{false})
        rbf_y_expr = build_function(rbf_y, [x, y]; expression = Val{false})
        rbf_z_expr = build_function(rbf_z, [x, y]; expression = Val{false})
        return (; rbf_expr, rbf_x_expr, rbf_y_expr, rbf_z_expr)
    end
end

function concrete_poly_flux_basis(poly, basis::RefPointData{NDIMS}) where {NDIMS}
    if NDIMS == 1
        # @polyvar x # diff wrt existing vars, new polyvar doesn't work
        poly_x = differentiate.(poly, poly[end].vars[1])
        f = StaticPolynomials.Polynomial.(poly)
        f_x = StaticPolynomials.Polynomial.(poly_x)
        poly_expr = PolynomialSystem(f...)
        poly_x_expr = PolynomialSystem(f_x...)
        return (; poly_expr, poly_x_expr)
    elseif NDIMS == 2
        # @polyvar x y
        poly_x = differentiate.(poly, poly[end].vars[1])
        poly_y = differentiate.(poly, poly[end].vars[2])
        f = StaticPolynomials.Polynomial.(poly)
        f_x = StaticPolynomials.Polynomial.(poly_x)
        f_y = StaticPolynomials.Polynomial.(poly_y)
        poly_expr = PolynomialSystem(f...)
        poly_x_expr = PolynomialSystem(f_x...)
        poly_y_expr = PolynomialSystem(f_y...)
        return (; poly_expr, poly_x_expr, poly_y_expr)
    elseif NDIMS == 3
        # @polyvar x y z
        poly_x = differentiate.(poly, poly[end].vars[1])
        poly_y = differentiate.(poly, poly[end].vars[2])
        poly_z = differentiate.(poly, poly[end].vars[3])
        f = StaticPolynomials.Polynomial.(poly)
        f_x = StaticPolynomials.Polynomial.(poly_x)
        f_y = StaticPolynomials.Polynomial.(poly_y)
        f_z = StaticPolynomials.Polynomial.(poly_z)
        poly_expr = PolynomialSystem(f...)
        poly_x_expr = PolynomialSystem(f_x...)
        poly_y_expr = PolynomialSystem(f_y...)
        poly_z_expr = PolynomialSystem(f_z...)
        return (; poly_expr, poly_x_expr, poly_y_expr, poly_z_expr)
    end
end

function rbf_block(rbf_expr, basis::RefPointData{NDIMS},
                   X::Vector{SVector{NDIMS, T}}) where {NDIMS, T}
    # Generate RBF Matrix for one interpolation point
    #
    # Inputs:   rbf_expr - RBF Function
    #           X - Input Point Set
    #
    # Outputs:  Φ - RBF Matrix Block

    m = lastindex(X)
    D = Array{SVector{NDIMS, T}, 2}(undef, m, m)

    for j in eachindex(X)
        for i in eachindex(X)
            D[i, j] = X[i] - X[j]
        end
    end

    return rbf_expr.(D)
end

function poly_block(poly_func, basis::RefPointData{NDIMS},
                    X::Vector{SVector{NDIMS, T}}) where {NDIMS, T}
    # Generate the polynomial basis block for one
    #  interpolation point
    #
    # Inputs:   F - StaticPolynomial Array
    #           X - Input Point Set
    #
    # Outputs:  P - Monomial Basis Block

    n = length(poly_func)
    m = lastindex(X)

    P = zeros(T, m, n)

    for i in eachindex(X)
        P[i, :] = StaticPolynomials.evaluate(poly_func, X[i])
    end

    return P
end

# Port of generator_operator from RBFD to generate Dx and Dy flux operators
# May move to separate file to load between types and rbfsolver
function compute_flux_operator(solver::RBFSolver,
                               domain::PointCloudDomain{NDIMS}) where {NDIMS}
    @unpack basis = solver
    @unpack rbf, poly = basis.f
    @unpack points, neighbors, num_points, num_neighbors = domain.pd

    rbf_func = concrete_rbf_flux_basis(rbf, basis)
    poly_func = concrete_poly_flux_basis(poly, basis)

    # Solve RBF interpolation system for all points
    for e in 1#eachelement(domain, solver)
        neighbor_idx = domain.pd.neighbors[e]
        X = domain.pd.points[neighbor_idx]
        R = rbf_block(rbf_func.rbf_expr, basis, X)
        P = poly_block(poly_func.poly_expr, basis, X)
        return (R, P)
    end
end
