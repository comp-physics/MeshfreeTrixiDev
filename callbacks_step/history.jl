function modify_cache!(source::T) where {T}
    # Fallback method that does nothing
end

function modify_cache!(source::SourceResidualViscosityTominec)
    # Modify the cache of SourceResidualViscosityTominec instances
    # Example: Updating a value in the cache
    source.cache.some_field = new_value
end
