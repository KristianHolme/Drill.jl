# ------------------------------------------------------------
# Entropy target
# ------------------------------------------------------------
abstract type AbstractEntropyTarget end

struct FixedEntropyTarget{T <: AbstractFloat} <: AbstractEntropyTarget
    target::T
end

struct AutoEntropyTarget <: AbstractEntropyTarget end

# ------------------------------------------------------------
# Entropy coefficient
# ------------------------------------------------------------
abstract type AbstractEntropyCoefficient end

struct FixedEntropyCoefficient{T <: AbstractFloat} <: AbstractEntropyCoefficient
    coef::T
end

@kwdef struct AutoEntropyCoefficient{T <: AbstractFloat, E <: AbstractEntropyTarget} <: AbstractEntropyCoefficient
    target::E = AutoEntropyTarget()
    initial_value::T = 1.0f0
end

Base.string(e::AutoEntropyCoefficient) = "AutoEntropyCoefficient(target=$(e.target), initial_value=$(e.initial_value))"
Base.string(e::FixedEntropyCoefficient) = "FixedEntropyCoefficient(coef=$(e.coef))"
