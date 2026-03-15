# BatchedCategorical distribution for efficient batched sampling

"""
    BatchedCategorical <: Distribution{Discrete, 1}

A distribution representing a batch of independent categorical distributions.
Useful for efficient batched sampling in reinforcement learning contexts.

# Fields
- `probs`: Matrix of probabilities where each column is a categorical distribution
           (size: num_classes × batch_size)

# Example
```julia
probs = [0.2 0.1 0.7; 0.3 0.5 0.2; 0.5 0.4 0.1]  # 3 classes, 3 batches
dist = BatchedCategorical(probs)
sample(rng, dist)  # Returns a one-hot encoded matrix (3×3)
```
"""
struct BatchedCategorical <: Distribution{Discrete, 1}
    probs::Matrix{Float32}
    
    function BatchedCategorical(probs::AbstractMatrix{T}) where {T<:Real}
        # Ensure probabilities are normalized along class dimension (rows)
        normalized_probs = probs ./ sum(probs, dims=1)
        new(float32.(normalized_probs))
    end
end

"""
    Drill.DrillDistributions.BatchedCategorical()
    
Constructor for BatchedCategorical distribution.
"""
BatchedCategorical() = BatchedCategorical(zeros(Float32, 0, 0))

"""
    rand(rng::AbstractRNG, d::BatchedCategorical, probs::AbstractMatrix)
    
Sample from a batched categorical distribution using the Gumbel-max trick.
Optimized for performance while maintaining Reactant compatibility.

# Arguments
- `rng`: Random number generator
- `d`: BatchedCategorical distribution (unused, for interface compatibility)
- `probs`: Matrix of probabilities (num_classes × batch_size)

# Returns
- One-hot encoded matrix of samples (num_classes × batch_size)
"""
function rand(rng::AbstractRNG, d::BatchedCategorical, probs::AbstractMatrix{T}) where {T<:AbstractFloat}
    # Optimized Gumbel-max trick for batched sampling
    # Reduce temporary allocations and improve numerical stability
    
    # Clamp probabilities to avoid log(0)
    epsval = eps(T)
    clamped_probs = clamp.(probs, epsval, one(T)-epsval)
    
    # Generate Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0,1)
    # More numerically stable: G = log(-log(U)) where U ~ Uniform(0,1) then negate
    U = rand(rng, T, size(probs))
    U = clamp.(U, epsval, one(T)-epsval)  # Clamp again for safety
    G = -log.(-log.(U))
    
    # Compute scores: log(probs) + Gumbel noise
    log_probs = log.(clamped_probs)
    scores = log_probs .+ G
    
    # Find max along class dimension (rows) for each batch element
    # Using findmax instead of maximum + equality comparison for better performance
    _, indices = findmax(scores, dims=1)
    
    # Convert to one-hot encoding efficiently
    num_classes, batch_size = size(probs)
    one_hot = zeros(T, num_classes, batch_size)
    
    # Use @inbounds for performance if indices are guaranteed valid
    @inbounds for b in 1:batch_size
        one_hot[indices[b], b] = one(T)
    end
    
    return one_hot
end

# For compatibility with the Distribution interface
function rand(rng::AbstractRNG, d::BatchedCategorical)
    error("BatchedCategorical requires probs argument for sampling. Use rand(rng, d, probs)")
end