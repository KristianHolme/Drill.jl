# Device transfer via adapt_structure (same API as device(data)).

# Mark RLCache as a leaf so fmap doesn't recurse into its fields.
# Our adapt_structure method below handles the actual device transfer.
isleaf(::RLCache) = true
isleaf(::PPOTrainState) = true
isleaf(::SACTrainState) = true

function adapt_structure(to::AbstractDevice, cache::RLCache)
    new_train_state = adapt(to, cache.train_state)
    return RLCache(
        cache.prob,
        cache.alg,
        cache.model,
        cache.adapter,
        new_train_state,
        cache.buffer,
        cache.logger,
        cache.rng,
        cache.verbose,
        cache.callbacks,
        cache.max_steps,
        cache.steps_taken,
        cache.gradient_updates,
        cache.ad_type,
        cache.retcode,
        cache.stats,
        cache.timer,
        nothing,
        cache.workspace,
    )
end

function adapt_structure(to::AbstractDevice, ts::PPOTrainState)
    return PPOTrainState(adapt(to, ts.ts))
end

function adapt_structure(to::AbstractDevice, ts::SACTrainState)
    return SACTrainState(
        adapt(to, ts.actor_ts),
        adapt(to, ts.critic_ts),
        adapt(to, ts.ent_ts),
        to(ts.target_parameters),
        to(ts.target_states),
    )
end
