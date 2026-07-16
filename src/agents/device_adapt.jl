# Device transfer via Adapt.adapt_structure (same API as device(data))
# Requires MLDataDevices.AbstractDevice and Adapt from the parent module.

# Mark Agent as a leaf so fmap doesn't recurse into its fields.
# Our adapt_structure method below handles the actual device transfer.
MLDataDevices.isleaf(::Agent) = true
MLDataDevices.isleaf(::PPOTrainState) = true
MLDataDevices.isleaf(::SACTrainState) = true

function Adapt.adapt_structure(to::MLDataDevices.AbstractDevice, agent::Agent)
    new_train_state = Adapt.adapt(to, agent.train_state)
    new_aux = Adapt.adapt(to, agent.aux)
    # Return a new Agent to avoid in-place modification
    return Agent(
        agent.layer,
        agent.algorithm,
        agent.action_adapter,
        new_train_state,
        agent.optimizer_type,
        agent.stats_window,
        agent.logger,
        agent.verbose,
        agent.rng,
        agent.stats,
        new_aux,
        nothing  # Reset cache
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

function Adapt.adapt_structure(::MLDataDevices.AbstractDevice, ::NoAux)
    return NoAux()
end
