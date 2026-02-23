"""
custom implementations of distributions. 

functions:
extend Random.rand
logpdf to get logprobs
entropy to get entropy
mode to get the mode

distributions:
Categorical

DiagGaussian
also used for action spaces with only one action element
    - mean is the action mean, doesnt need to be a vector, to preserve action shape
    - std, same shape as mean
"""


struct Categorical{V <: AbstractVector{<:Real}} <: AbstractDiscreteDistribution
    probabilities::V
    start::Integer
    function Categorical(probs::V, start::Integer = 1) where {V <: AbstractVector{<:Real}}
        @assert sum(probs) ≈ 1 "Sum of probabilities must be 1"
        return new{V}(probs, start)
    end
end

function logpdf(d::Categorical, x::AbstractArray{<:Integer})
    @assert length(x) == 1 "Categorical distribution only supports single actions"
    return logpdf(d, x[1])
end

function logpdf(d::Categorical, x::Integer)
    return log(d.probabilities[x - d.start + 1])
end

function entropy(d::Categorical)
    return -sum(d.probabilities .* log.(d.probabilities))
end

function mode(d::Categorical)
    return argmax(d.probabilities) + d.start - 1
end


function Random.rand(rng::AbstractRNG, d::Categorical)
    cumulative_probs = cumsum(d.probabilities)
    u = rand(rng)
    idx = findfirst(cumulative_probs .>= u)
    return idx + d.start - 1
end

Random.rand(rng::AbstractRNG, d::Categorical, n::Integer) = [rand(rng, d) for _ in 1:n]
