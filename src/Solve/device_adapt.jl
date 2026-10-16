# Device transfer via adapt_structure (same API as device(data)).

# Mark RLCache as a leaf so fmap doesn't recurse into its fields.
# Our adapt_structure method below handles the actual device transfer.
isleaf(::RLCache) = true

function adapt_structure(to::AbstractDevice, cache::RLCache)
    new_learner = to(cache.learner)
    return RLCache(
        cache.prob,
        cache.alg,
        cache.model,
        cache.adapter,
        new_learner,
        cache.buffer,
        cache.logger,
        cache.rng,
        cache.verbosity,
        cache.progress_meter,
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
