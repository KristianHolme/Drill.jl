"""
Diagonal covariance Gaussian distribution (arbitrary shape)
"""
struct DiagGaussian{T <: Real, M <: AbstractArray{T}, S <: AbstractArray{T}} <: AbstractContinuousDistribution
    mean::M
    log_std::S
    function DiagGaussian(mean::M, log_std::S) where {T <: Real, M <: AbstractArray{T}, S <: AbstractArray{T}}
        @assert size(mean) == size(log_std) "Mean and log_std must have the same shape"
        return new{T, M, S}(mean, log_std)
    end
end

function Random.rand(rng::AbstractRNG, d::DiagGaussian{T, M, S}) where {T, M, S}
    # Ensure the random values are of type T for type stability
    noise = randn(rng, T, size(d.mean))
    return d.mean .+ exp.(d.log_std) .* noise
end

function Random.rand(rng::AbstractRNG, d::DiagGaussian, n::Integer)
    return [rand(rng, d) for _ in 1:n]
end

const log2π = log(2π)

function logpdf(d::DiagGaussian{T, M, S}, x::AbstractArray{T}) where {T, M, S}
    k = length(d.mean)
    log_std_sum = sum(d.log_std)

    diff = x - d.mean
    var_inv = exp.(-T(2) * d.log_std)
    diff_squared_sum = sum(abs2.(diff) .* var_inv)

    result = -T(0.5) * (T(2) * log_std_sum + diff_squared_sum + k * T(log2π))

    return result
end

function entropy(d::DiagGaussian{T, M, S}) where {T, M, S}
    k = length(d.mean)
    log_std_sum = sum(d.log_std)
    result = T(0.5) * k * T(1 + log2π) + log_std_sum
    return result
end

function mode(d::DiagGaussian)
    return d.mean
end

# =============================================================================
# Batched DiagGaussian (empty struct for dispatch)
# =============================================================================

struct BatchedDiagGaussian <: AbstractContinuousDistribution
end

function Random.rand(rng::AbstractRNG, ::BatchedDiagGaussian, mean::AbstractArray{T}, log_std::AbstractArray{T}) where {T}
    noise = randn(rng, T, size(mean))
    return mean .+ exp.(log_std) .* noise
end

function mode(::BatchedDiagGaussian, mean::AbstractArray)
    return mean
end

"""
Batch sum over all dimensions except the last one, return 1×batch_size array.
"""
function _dsum(x::AbstractArray{T, N}, dims) where {T, N}
    dims_drop = dims[1:(end - 1)]
    return dropdims(sum(x; dims = dims), dims = dims_drop)
end

function logpdf(::BatchedDiagGaussian, x::AbstractArray{T, N}, mean::AbstractArray{T, N}, log_std::AbstractArray{T, N}) where {T, N}
    k = prod(size(mean)[1:(end - 1)])
    non_batch_dims = ntuple(i -> i, N - 1)
    log_std_sum = _dsum(log_std, non_batch_dims)
    diff = x - mean
    var_inv = exp.(-T(2) * log_std)
    diff_squared_sum = _dsum(abs2.(diff) .* var_inv, non_batch_dims)
    results = -T(0.5) * (T(2) * log_std_sum .+ diff_squared_sum .+ k * T(log2π))
    return results
end

function entropy(::BatchedDiagGaussian, mean::AbstractArray{T, N}, log_std::AbstractArray{T, N}) where {T, N}
    k = prod(size(mean)[1:(end - 1)])
    non_batch_dims = ntuple(i -> i, N - 1)
    log_std_sum = _dsum(log_std, non_batch_dims)
    results = T(0.5) * (T(k) * T(1 + log2π)) .+ log_std_sum
    return results
end
