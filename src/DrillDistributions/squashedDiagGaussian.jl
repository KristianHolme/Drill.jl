"""
    SquashedDiagGaussian(mean::AbstractArray, log_std::AbstractArray; eps::Real=1e-6) -> SquashedDiagGaussian

A squashed diagonal Gaussian distribution.

# Arguments
- `mean::AbstractArray`: The mean of the Gaussian distribution.
- `log_std::AbstractArray`: The log of the standard deviation of the Gaussian distribution.
- `eps::Real`: The epsilon value to avoid numerical instability when inverting the tanh function.
"""
struct SquashedDiagGaussian{T <: Real, M <: AbstractArray{T}, S <: AbstractArray{T}} <: AbstractContinuousDistribution
    DiagGaussian::DiagGaussian{T, M, S}
    epsilon::T
    @inline function SquashedDiagGaussian(mean::M, log_std::S, eps::T) where {T <: Real, M <: AbstractArray{T}, S <: AbstractArray{T}}
        @assert size(mean) == size(log_std) "Mean and log_std must have the same shape"
        return new{T, M, S}(DiagGaussian(mean, log_std), eps)
    end
    @inline function SquashedDiagGaussian(mean::M, log_std::S) where {T <: Real, M <: AbstractArray{T}, S <: AbstractArray{T}}
        @assert size(mean) == size(log_std) "Mean and log_std must have the same shape"
        return new{T, M, S}(DiagGaussian(mean, log_std), T(1.0e-6))
    end
end

function Random.rand(rng::AbstractRNG, d::SquashedDiagGaussian)
    sample = rand(rng, d.DiagGaussian)
    return tanh.(sample)
end

function Random.rand(rng::AbstractRNG, d::SquashedDiagGaussian, n::Integer)
    return [rand(rng, d) for _ in 1:n]
end

function logpdf(d::SquashedDiagGaussian{T, M, S}, x::AbstractArray{T}) where {T <: Real, M, S}
    gaussian_action = atanh.(clamp.(x, Ref(-T(1) + d.epsilon::T), Ref(T(1) - d.epsilon::T)))
    gaussian_logpdf = logpdf(d.DiagGaussian, gaussian_action)
    # More numerically stable formula: 2*(log(2) - x - softplus(-2*x)) instead of log(1 - tanh(x)^2)
    # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    #TODO: type stability, getting Float64
    #TODO: make test for this
    #TODO: fix runtime dispatch here
    correction = T(2) * (log(T(2)) .- gaussian_action - Lux.softplus.(-T(2) * gaussian_action))
    squashed_logpdf = gaussian_logpdf - sum(correction)
    return squashed_logpdf::T
end

#not implemented: entropy, as its implemented directly in the loss

function mode(d::SquashedDiagGaussian)
    return tanh.(d.DiagGaussian.mean)
end
