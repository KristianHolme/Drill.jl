"""
Batched Categorical distribution (empty struct for dispatch).
Returns dense one-hot matrices of shape (num_classes, batch_size) with eltype matching probs.
Probs shape: (num_classes, batch_size). Kernel-friendly (tensor ops only) for Reactant.
"""
struct BatchedCategorical <: AbstractDiscreteDistribution
end

function Random.rand(rng::AbstractRNG, ::BatchedCategorical, probs::AbstractMatrix{T}) where {T}
    epsval = eps(T)
    U = rand(rng, T, 1, size(probs, 2))
    U = @. clamp(U, epsval, one(T) - epsval)
    cdf = cumsum(probs; dims = 1)
    return @. ifelse((cdf >= U) & (cdf - probs < U), one(T), zero(T))
end

function mode(::BatchedCategorical, probs::AbstractMatrix{T}) where {T}
    # Tie-break so exactly one 1 per column (deterministic: smallest index wins)
    num_classes = size(probs, 1)
    tie_breaker = eps(T) .* (1:num_classes)
    scores = probs .+ reshape(tie_breaker, :, 1)
    max_scores = reshape(maximum(scores, dims = 1), (1, size(probs, 2)))
    one_hot = (scores .== max_scores)
    return T.(one_hot)
end

function logpdf(::BatchedCategorical, x::AbstractMatrix, probs::AbstractMatrix)
    return log.(sum(probs .* x, dims = 1))
end

function entropy(::BatchedCategorical, probs::AbstractMatrix)
    probs_clamped = clamp.(probs, 0, 1)
    plogp = probs_clamped .* log.(probs_clamped)
    return -sum(plogp, dims = 1)
end
