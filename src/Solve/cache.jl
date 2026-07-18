mutable struct RLCache{P, A, M, AD, TS, B, L, R, C, ST, TO, PM}
    prob::P
    alg::A
    model::M
    adapter::AD
    train_state::TS
    buffer::B
    logger::L
    rng::R
    verbosity::Verbosity
    progress_meter::PM
    callbacks::C
    max_steps::Int
    steps_taken::Int
    gradient_updates::Int
    ad_type
    retcode::ReturnCode.T
    stats::ST
    timer::TO
    inference_cache::Any
    workspace::Dict{Symbol, Any}
end

function parameters(cache::RLCache)
    return parameters(cache.train_state)
end

function states(cache::RLCache)
    return states(cache.train_state)
end

function set_states!(cache::RLCache, st)
    set_states!(cache.train_state, st)
    return cache
end

function invalidate_cache!(cache::RLCache)
    cache.inference_cache = nothing
    return cache
end

function steps_taken(cache::RLCache)
    return cache.steps_taken
end

function add_steps!(cache::RLCache, n::Integer)
    cache.steps_taken += n
    return cache.steps_taken
end

function add_gradient_update!(cache::RLCache)
    cache.gradient_updates += 1
    return cache.gradient_updates
end

function ensure_stat!(cache::RLCache, key::Symbol, ::Type{T} = Float32) where {T}
    if !haskey(cache.stats, key)
        cache.stats[key] = T[]
    end
    return cache.stats[key]
end
