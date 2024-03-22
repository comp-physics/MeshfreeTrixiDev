# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    HistoryCallback(; cfl=1.0)

Update solution time history according to a spline interpolation with 
reconstruction order `polydeg`. Utilized to approximate residual 
for targeted residual based viscosity.
"""
mutable struct HistoryCallback{RealT}
    polydeg::RealT
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:HistoryCallback})
    @nospecialize cb # reduce precompilation time

    history_callback = cb.affect!
    @unpack polydeg = history_callback
    print(io, "HistoryCallback(polydeg=", polydeg, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:HistoryCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        history_callback = cb.affect!

        setup = [
            "Reconstruction Order" => history_callback.polydeg
        ]
        summary_box(io, "HistoryCallback", setup)
    end
end

function HistoryCallback(; cfl::Real = 1.0)
    history_callback = HistoryCallback()

    DiscreteCallback(history_callback, history_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: HistoryCallback}
    cb.affect!(integrator)
end

# this method is called to determine whether the callback should be activated
function (history_callback::HistoryCallback)(u, t, integrator)
    return true
end

# This method is called as callback during the time integration.
@inline function (history_callback::HistoryCallback)(integrator)
    t = integrator.t
    u_ode = integrator.u
    semi = integrator.p
    @unpack polydeg = history_callback

    # Dispatch based on semidiscretization
    @trixi_timeit timer() "update history" update_history!(semi, u_ode, t, polydeg,
                                                           integrator)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

function update_history!(semi, u, t, polydeg, integrator)
    @unpack source_terms = semi

    # If source includes time history cache, update
    # otherwise no-op
    for source in values(source_terms)
        modify_cache!(source, u, t, polydeg, integrator)
    end
end

function modify_cache!(source::T, u, t, polydeg, integrator) where {T}
    # Fallback method that does nothing
end

function modify_cache!(source::SourceResidualViscosityTominec, u, t, polydeg,
                       integrator)
    # May need access to integrator to get count of timesteps 
    # since first few iterations can only support lower order
    @unpack time_history, sol_history, approx_du, time_weights = source.cache
    @unpack success_iter, iter, saveiter, saveiter_dense, last_stepfail, accept_step = integrator

    shift_soln_history!(time_history, sol_history, t, u)
    update_approx_du!(approx_du, time_weights, time_history, sol_history, success_iter,
                      polydeg)
end

function shift_soln_history!(time_history, sol_history, t, u)
    # Assuming sol_history[:, 1] is the most recent and sol_history[:, end] is the oldest
    time_history[2:end] .= time_history[1:(end - 1)]
    time_history[1] = t
    sol_history[:, 2:end] .= sol_history[:, 1:(end - 1)]
    sol_history[:, 1] .= u
end

function update_approx_du!(approx_du, time_weights, time_history, sol_history,
                           success_iter, polydeg)
    if success_iter == 0
        # approx_du .= 0.0
        set_to_zero!(approx_du)
    elseif success_iter == 1
        time_deriv_weights!(@view(time_weights[1:2]),
                            @view(time_history[1:2]))
        # @. approx_du = (1 / Δt) * (u_min1 - u_min2)
        @. approx_du = time_weights[end] * sol_history[:, 1] +
                       time_weights[end - 1] * sol_history[:, 2]
    elseif success_iter == 2
        time_deriv_weights!(@view(time_weights[1:3]),
                            @view(time_history[1:3]))
        # @. approx_du = (1 / Δt) * (3 / 2) * (u_min1 - (4 / 3) * u_min2 + (1 / 3) * u_min3)
        @. approx_du = time_weights[end] * sol_history[:, 1] +
                       time_weights[end - 1] * sol_history[:, 2] +
                       time_weights[end - 2] * sol_history[:, 3]
    elseif success_iter == 3
        time_deriv_weights!(@view(time_weights[1:4]),
                            @view(time_history[1:4]))
        # @. approx_du = (1 / Δt) * (11 / 6) * (u_min1 - (18 / 11) * u_min2 + (9 / 11) * u_min3 - (2 / 11) * u_min4)
        @. approx_du = time_weights[1] * sol_history[:, 1] +
                       time_weights[2] * sol_history[:, 2] +
                       time_weights[3] * sol_history[:, 3] +
                       time_weights[4] * sol_history[:, 4]
    else
        time_deriv_weights!(@view(time_weights[1:5]),
                            @view(time_history[1:5]))
        # @. approx_du = (1 / Δt) * (25 / 12) * (u_min1 - (48 / 25) * u_min2 + (36 / 25) * u_min3 - (16 / 25) * u_min4 + (3 / 25) * u_min5)
        @. approx_du = time_weights[1] * sol_history[:, 1] +
                       time_weights[2] * sol_history[:, 2] +
                       time_weights[3] * sol_history[:, 3] +
                       time_weights[4] * sol_history[:, 4] +
                       time_weights[5] * sol_history[:, 5]
        # @printf("Full reconstruction, max(approx_du) = %f, min(approx_du) = %f \n", maximum(approx_du), minimum(approx_du))
        # if maximum(approx_du) < 1e-10
        #     @printf("time_history: %f, %f, %f, %f, %f \n\n", time_history[end-4], time_history[end-3], time_history[end-2], time_history[end-1], time_history[end])
        # end
    end

    return nothing
end

function time_deriv_weights!(w, t)
    #Input: a vector t, where t(i) is time at which the solution is available.
    # Output: a vector w, where each w(i) is used to multiply u|_{t(i)} in order
    # ... to get a derivative at t(end).
    # Usage: d/dt u(t_end) = w(end)*u(end) + w(end-1)*u(end-1) + ... + w(1)*u(1),
    # ... where t_end is the time at which the last solution point is available.
    # From Tominec
    scale = 1 / maximum(abs.(t))
    t_ = t .* scale
    t_eval = t_[1] # The derivative should be evaluated at t(end).
    # Construct the polynomial basis, and differentiate it in a point t_eval.
    A = zeros(size(t_, 1), size(t_, 1))
    b_t = zeros(1, size(t_, 1))
    for k in 1:length(t)
        A[:, k] = t_ .^ (k - 1)
        b_t[k] = (k - 1) * t_eval .^ (k - 2)
    end
    # w .= scale .* (b_t / A)
    w .= scale .* (A' \ b_t')

    return nothing
end

# function update_soln_hist!(integrator, p, t, u)
#     # Extract history and state
#     @unpack success_iter, iter, saveiter, saveiter_dense, last_stepfail, accept_step = integrator
#     @unpack counter, approx_du, time_history, u_min1, u_min2, u_min3, u_min4, u_min5, u_min6, time_weight = p.time_cache
#     if success_iter > counter[1] # save history
#         time_history[1:(end - 1)] .= time_history[2:end]
#         time_history[end] = t
#         # counter_bool .= true
#         recursivecopy!(u_min6, u_min5) #u_min6 .= u_min5 # Update with recursivecopy!
#         recursivecopy!(u_min5, u_min4) #u_min5 .= u_min4
#         recursivecopy!(u_min4, u_min3) #u_min4 .= u_min3
#         recursivecopy!(u_min3, u_min2) #u_min3 .= u_min2
#         recursivecopy!(u_min2, u_min1) #u_min2 .= u_min1
#         recursivecopy!(u_min1, u) #u_min1 .= u
#         # Update approx_du
#         # time_deriv_weights!(time_weight, time_history)
#         update_ddt!(approx_du, time_weight, time_history, counter[1], u_min1, u_min2, u_min3,
#                     u_min4, u_min5)
#         # if p.operator_cache.c_rv[1] > 1.0
#         #     @. p.operator_cache.c_rv = p.operator_cache.c_rv - 0.001
#         #     # @printf("c_rv = %f \n", p.operator_cache.c_rv[1])
#         # end
#     elseif success_iter == counter[1] && counter[1] != 0
#         p.time_cache.counter[1] = p.time_cache.counter[1] - 1
#         time_history[end] = t
#         recursivecopy!(u_min1, u) #u_min1 .= u
#         # Update approx_du
#         # time_deriv_weights!(time_weight, time_history)
#         update_ddt!(approx_du, time_weight, time_history, counter[1], u_min1, u_min2, u_min3,
#                     u_min4, u_min5)
#         # @. p.operator_cache.c_rv = p.operator_cache.c_rv + 0.01
#     end

#     return nothing
# end

function update_ddt!(approx_du, w, t, count, u_min1, u_min2, u_min3, u_min4, u_min5)
    # Determine if step is updated or if we are still in the same step
    # i.e. Do all the solutions have to be updated or only the most recent one if smaller time step is used

    if count == 0
        approx_du .= 0.0
    elseif count == 1
        time_deriv_weights!(@view(w[(end - 1):end]),
                            @view(t[(end - 1):end]))
        # @. approx_du = (1 / Δt) * (u_min1 - u_min2)
        @inbounds @. approx_du = w[end] * u_min1 + w[end - 1] * u_min2
    elseif count == 2
        time_deriv_weights!(@view(w[(end - 2):end]),
                            @view(t[(end - 2):end]))
        # @. approx_du = (1 / Δt) * (3 / 2) * (u_min1 - (4 / 3) * u_min2 + (1 / 3) * u_min3)
        @inbounds @. approx_du = w[end] * u_min1 + w[end - 1] * u_min2 +
                                 w[end - 2] * u_min3
    elseif count == 3
        time_deriv_weights!(@view(w[(end - 3):end]),
                            @view(t[(end - 3):end]))
        # @. approx_du = (1 / Δt) * (11 / 6) * (u_min1 - (18 / 11) * u_min2 + (9 / 11) * u_min3 - (2 / 11) * u_min4)
        @inbounds @. approx_du = w[end] * u_min1 + w[end - 1] * u_min2 +
                                 w[end - 2] * u_min3 +
                                 w[end - 3] * u_min4
    else
        time_deriv_weights!(@view(w[(end - 4):end]),
                            @view(t[(end - 4):end]))
        # @. approx_du = (1 / Δt) * (25 / 12) * (u_min1 - (48 / 25) * u_min2 + (36 / 25) * u_min3 - (16 / 25) * u_min4 + (3 / 25) * u_min5)
        @inbounds @. approx_du = w[end] * u_min1 + w[end - 1] * u_min2 +
                                 w[end - 2] * u_min3 +
                                 w[end - 3] * u_min4 + w[end - 4] * u_min5
        # @printf("Full reconstruction, max(approx_du) = %f, min(approx_du) = %f \n", maximum(approx_du), minimum(approx_du))
        # if maximum(approx_du) < 1e-10
        #     @printf("time_history: %f, %f, %f, %f, %f \n\n", t[end-4], t[end-3], t[end-2], t[end-1], t[end])
        # end
    end

    return nothing
end
end # @muladd