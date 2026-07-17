# Algorithm traits and adapter selection

"""
    action_adapter(alg, action_space) -> AbstractActionAdapter

Return the action adapter to convert policy-space actions to env-space actions
for the given algorithm and action space. Algorithms should extend this.
"""
function action_adapter(alg::AbstractAlgorithm, action_space::AbstractSpace)
    error("action_adapter not implemented for $(typeof(alg)) with $(typeof(action_space))")
end

# Capability traits (defaults)
has_twin_critics(::AbstractAlgorithm) = false
has_target_networks(::AbstractAlgorithm) = false
has_entropy_tuning(::AbstractAlgorithm) = false

"""
    uses_replay(alg) -> Bool

Whether the algorithm uses a replay buffer for training.
"""
uses_replay(::AbstractAlgorithm) = false

"""
    critic_type(alg) -> CriticType

Return the critic type preferred by the algorithm (e.g., `VCritic()` or `QCritic()`).
"""
critic_type(::AbstractAlgorithm) = VCritic()
