# Device transfer via Adapt.adapt_structure (same API as device(data))
# Requires MLDataDevices.AbstractDevice and Adapt from the parent module.

# Mark RLCache as a leaf so fmap doesn't recurse into its fields.
# Our adapt_structure method below handles the actual device transfer.
MLDataDevices.isleaf(::RLCache) = true
MLDataDevices.isleaf(::PPOTrainState) = true
MLDataDevices.isleaf(::SACTrainState) = true

function Adapt.adapt_structure(to::MLDataDevices.AbstractDevice, cache::RLCache)
    new_train_state = Adapt.adapt(to, cache.train_state)
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
        cache.optimizer_type,
        cache.workspace,
    )
end

function Adapt.adapt_structure(to::MLDataDevices.AbstractDevice, ts::PPOTrainState)
    return PPOTrainState(Adapt.adapt(to, ts.ts))
end

function Adapt.adapt_structure(to::MLDataDevices.AbstractDevice, ts::SACTrainState)
    return SACTrainState(
        Adapt.adapt(to, ts.actor_ts),
        Adapt.adapt(to, ts.critic_ts),
        Adapt.adapt(to, ts.ent_ts),
        to(ts.target_parameters),
        to(ts.target_states),
    )
end
