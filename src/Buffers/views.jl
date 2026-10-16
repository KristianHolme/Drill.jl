# Transition and episode views of buffer storage. The buffers store one array per field
# (structure of arrays); these views present the same memory as arrays of transitions.
# Nothing is copied: reading a view reads the buffer, and writing to it writes the buffer.

# One element per transition: the last dimension of `x` indexes transitions.
_per_transition(x::AbstractVector) = x
_per_transition(x::AbstractArray) = eachslice(x; dims = ndims(x))

"""
    transitions(buffer::RolloutBuffer) -> StructArray

All transitions of `buffer` as a `StructArray`, in storage order (step-major, see
[`RolloutBuffer`](@ref)). Column `c` is `transitions(buffer).c`, a view of `buffer.c`;
element `i` is a `NamedTuple` with one field per column. The columns are `observations`,
`actions`, `rewards`, `logprobs`, `values`, `advantages`, `returns`, `terminateds`,
`truncateds` and `bootstrap_values`. An observation or action of a transition is a view
of its slice of `buffer.observations` or `buffer.actions`.
"""
function transitions(buffer::RolloutBuffer)
    return StructArray(
        (;
            observations = _per_transition(buffer.observations),
            actions = _per_transition(buffer.actions),
            buffer.rewards,
            buffer.logprobs,
            buffer.values,
            buffer.advantages,
            buffer.returns,
            buffer.terminateds,
            buffer.truncateds,
            buffer.bootstrap_values,
        )
    )
end

"""
    env_transitions(buffer::RolloutBuffer, env::Int) -> StructArray

The `n_steps` transitions of env `env` in time order, as a view of
[`transitions(buffer)`](@ref transitions). Episode boundaries are marked by
`terminateds` and `truncateds`; see [`episodes`](@ref) for the transitions split by
episode.
"""
function env_transitions(buffer::RolloutBuffer, env::Int)
    1 <= env <= buffer.n_envs || throw(ArgumentError("env must be in 1:$(buffer.n_envs), got $env"))
    return view(transitions(buffer), env:buffer.n_envs:length(buffer))
end

"""
    episodes(buffer::RolloutBuffer) -> Vector{<:StructArray}

The transitions of `buffer` split into episodes, each a view of
[`transitions(buffer)`](@ref transitions) in time order. Episodes are ordered by env,
then by time. An episode ends at a `terminateds` or `truncateds` flag, or at the last
step of the rollout; in the last case neither flag is set on its final transition, and
the episode continues in the next rollout. Every transition is in exactly one episode.
"""
function episodes(buffer::RolloutBuffer)
    all_transitions = transitions(buffer)
    n_envs, n = buffer.n_envs, length(buffer)
    result = typeof(view(all_transitions, 1:n_envs:1))[]
    for env in 1:n_envs
        first_index = env
        for i in env:n_envs:n
            if buffer.terminateds[i] || buffer.truncateds[i] || i > n - n_envs
                push!(result, view(all_transitions, first_index:n_envs:i))
                first_index = i + n_envs
            end
        end
    end
    return result
end

"""
    transitions(buffer::ReplayBuffer) -> StructArray

The stored transitions of `buffer` as a `StructArray`, oldest first. The columns are
`observations`, `actions`, `rewards`, `terminated`, `truncated` and `next_observations`,
views of the fields of the same names.

The replay buffer does not record which env a transition came from, so it has no episode
view.
"""
function transitions(buffer::ReplayBuffer)
    all_slots = StructArray(
        (;
            observations = _per_transition(buffer.observations),
            actions = _per_transition(buffer.actions),
            buffer.rewards,
            buffer.terminated,
            buffer.truncated,
            next_observations = _per_transition(buffer.next_observations),
        )
    )
    oldest_first = if isfull(buffer)
        vcat(buffer.position:capacity(buffer), 1:(buffer.position - 1))
    else
        collect(1:length(buffer))
    end
    return view(all_slots, oldest_first)
end
