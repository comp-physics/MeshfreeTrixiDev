# Multiple calc_sources! to resolve method ambiguities
function calc_sources!(du, u, t, source_terms::Nothing,
                       domain, equations, solver::PointCloudSolver, cache)
    nothing
end

# uses quadrature + projection to compute source terms.
function calc_sources!(du, u, t, source_terms,
                       domain, equations, solver::PointCloudSolver, cache)
    for source in values(source_terms)
        source(du, u, t, domain, equations, solver, cache)
    end
end

# Instead of dedicated function, use callable struct 
# for each specific source term
# function calc_single_source!()
# end