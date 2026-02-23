"""
Batched Categorical distribution (empty struct for dispatch).
Uses OneHotVectors for actions. Probs shape: (num_classes, batch_size).
"""
struct BatchedCategorical <: AbstractDiscreteDistribution
end

function Random.rand(rng::AbstractRNG, ::BatchedCategorical, probs::AbstractMatrix)
    num_classes, batch_size = size(probs)
    indices = Vector{Int}(undef, batch_size)
    for b in 1:batch_size
        cum = cumsum(probs[:, b])
        u = rand(rng)
        idx = findfirst(cum .>= u)
        indices[b] = idx === nothing ? num_classes : idx
    end
    return OneHotArrays.onehotbatch(indices, 1:num_classes)
end

function mode(::BatchedCategorical, probs::AbstractMatrix)
    idx = argmax(probs; dims = 1)
    num_classes, batch_size = size(probs)
    return OneHotArrays.onehotbatch([idx[i][1] for i in 1:batch_size], 1:num_classes)
end

function logpdf(::BatchedCategorical, x::AbstractMatrix, probs::AbstractMatrix)
    return log.(sum(probs .* x, dims = 1))
end

function entropy(::BatchedCategorical, probs::AbstractMatrix)
    probs_clamped = clamp.(probs, 0, 1)
    plogp = probs_clamped .* log.(probs_clamped)
    return -sum(plogp, dims = 1)
end
