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

function rbf_block(rbf_func, basis::RefPointData{NDIMS}; N::Int) where {NDIMS}
end

function poly_block(poly_func, basis::RefPointData{NDIMS}; N::Int) where {NDIMS}
end

# Port of generator_operator from RBFD to generate Dx and Dy flux operators
# May move to separate file to load between types and rbfsolver
function compute_flux_operator(solver::RBFSolver,
                               domain::PointCloudDomain{NDIMS}) where {NDIMS}
    @unpack basis = solver
    @unpack rbf, poly = basis.f

    rbf_func = concrete_rbf_flux_basis(rbf, basis)
    poly_func = concrete_poly_flux_basis(poly, basis)
end
