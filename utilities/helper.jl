function show_ft(nt::NamedTuple)
    for (name, value) in pairs(nt)
        println("$(name): $(typeof(value))")
    end
end
