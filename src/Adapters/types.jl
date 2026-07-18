# ------------------------------------------------------------
# Action Adapters (policy-space <-> env-space)
# ------------------------------------------------------------

"""
    AbstractActionAdapter

Maps between policy outputs and environment actions (see `to_env`, `from_env`); concrete types include `ClampAdapter`, `TanhScaleAdapter`, and `DiscreteAdapter`.
"""
abstract type AbstractActionAdapter end

"""
    to_env(adapter, policy_action, space::AbstractSpace)

Convert an action from the policy/model's action domain to the environment's
action space. Called right before stepping the environment.
"""
function to_env end

"""
    from_env(adapter, env_action, space::AbstractSpace)

Optionally convert an environment action back to the policy/model's action
domain. Useful for some off-policy training flows. Default: identity where
appropriate.
"""
function from_env end
#TODO: move these to not be in the interface file?
# Concrete adapter types (behavior selected by algorithm)
struct ClampAdapter <: AbstractActionAdapter end          # e.g., PPO with Box
struct TanhScaleAdapter <: AbstractActionAdapter end      # e.g., SAC with Box
struct DiscreteAdapter <: AbstractActionAdapter end       # Discrete actions

# Fallbacks to surface missing implementations early
to_env(::AbstractActionAdapter, action, space) = error("to_env not implemented for $(typeof(space)) with $(typeof(action))")
from_env(::AbstractActionAdapter, action, space) = action
