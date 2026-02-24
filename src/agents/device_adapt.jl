# Device transfer via Adapt.adapt_structure (same API as device(data))
# Requires MLDataDevices.AbstractDevice and Adapt from the parent module.

function Adapt.adapt_structure(to::MLDataDevices.AbstractDevice, agent::Agent)
    new_train_state = Adapt.adapt(to, agent.train_state)
    new_aux = Adapt.adapt(to, agent.aux)
    agent = @set agent.train_state = new_train_state
    agent = @set agent.aux = new_aux
    return agent
end

function Adapt.adapt_structure(to::MLDataDevices.AbstractDevice, aux::QAux)
    new_Q_target_parameters = to(aux.Q_target_parameters)
    new_Q_target_states = to(aux.Q_target_states)
    new_ent_train_state = Adapt.adapt(to, aux.ent_train_state)
    return QAux(new_Q_target_parameters, new_Q_target_states, new_ent_train_state)
end

function Adapt.adapt_structure(::MLDataDevices.AbstractDevice, ::NoAux)
    return NoAux()
end
