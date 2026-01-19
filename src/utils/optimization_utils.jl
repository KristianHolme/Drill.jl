# Optimization utility functions (used by SAC)

function polyak_update!(target::AbstractArray{T}, source::AbstractArray{T}, tau::T) where {T <: AbstractFloat}
    target .= tau .* source .+ (one(T) - tau) .* target
    return nothing
end

function polyak_update!(target::ComponentArray{T}, source::ComponentArray{T}, tau::T) where {T <: AbstractFloat}
    for key in keys(target)
        target[key] .= tau .* source[key] .+ (one(T) - tau) .* target[key]
    end
    return nothing
end

function polyak_update!(target::NamedTuple, source::NamedTuple, tau::T) where {T <: AbstractFloat}
    # Update arrays in-place by traversing both structures together
    for key in keys(target)
        if haskey(source, key)
            t_val = target[key]
            s_val = source[key]
            if t_val isa AbstractArray && s_val isa AbstractArray
                t_val .= tau .* s_val .+ (one(T) - tau) .* t_val
            elseif t_val isa NamedTuple && s_val isa NamedTuple
                polyak_update!(t_val, s_val, tau)
            end
        end
    end
    return nothing
end

function merge_params(a1::ComponentArray, a2::ComponentArray)
    a3 = copy(a1)
    for key in keys(a2)
        a3[key] = a2[key]
    end
    return a3
end

function merge_params(a1::NamedTuple, a2::NamedTuple)
    # Merge a2 into a1, creating a new NamedTuple
    # For arrays, we need to deepcopy to avoid aliasing
    result_parts = []
    for key in keys(a1)
        if haskey(a2, key)
            v1 = a1[key]
            v2 = a2[key]
            if v1 isa AbstractArray && v2 isa AbstractArray
                push!(result_parts, key => deepcopy(v2))
            elseif v1 isa NamedTuple && v2 isa NamedTuple
                push!(result_parts, key => merge_params(v1, v2))
            else
                push!(result_parts, key => deepcopy(v2))
            end
        else
            push!(result_parts, key => a1[key])
        end
    end
    return NamedTuple(result_parts)
end

"""
    nested_norm(ps, T)

Calculate the L2 norm of nested parameter structures (tuples/NamedTuples) by traversing
all arrays using Functors.fmap.

# Arguments
- `ps`: Nested parameter structure (typically from Lux.jl)
- `T`: Type for zero initialization (e.g., Float32)

# Returns
- The L2 norm of all arrays in the structure
"""
function nested_norm(ps, T::Type{<:AbstractFloat})
    s = Ref(zero(T))
    fmap(ps) do x
        if x isa AbstractArray
            s[] += sum(abs2, x)
        end
        return x
    end
    return sqrt(s[])
end

"""
    nested_scale!(ps, max_norm, norm_val)

Scale all arrays in a nested parameter structure in-place to have a maximum norm.

# Arguments
- `ps`: Nested parameter structure to scale (modified in-place)
- `max_norm`: Maximum allowed norm
- `norm_val`: Current norm of the structure (should be computed with nested_norm)

# Returns
- The modified parameter structure `ps`
"""
function nested_scale!(ps, max_norm::T, norm_val::T) where {T <: AbstractFloat}
    scale = max_norm / norm_val
    fmap(ps) do x
        if x isa AbstractArray
            x .= x .* scale
        end
        return x
    end
    return ps
end

"""
    nested_has_nan(ps)

Check if any array in a nested parameter structure contains NaN values.

# Arguments
- `ps`: Nested parameter structure (typically from Lux.jl)

# Returns
- `true` if any array contains NaN, `false` otherwise
"""
function nested_has_nan(ps)
    has_nan = Ref(false)
    fmap(ps) do x
        if x isa AbstractArray
            if any(isnan, x)
                has_nan[] = true
            end
        end
        return x
    end
    return has_nan[]
end

"""
    nested_has_inf(ps)

Check if any array in a nested parameter structure contains Inf values.

# Arguments
- `ps`: Nested parameter structure (typically from Lux.jl)

# Returns
- `true` if any array contains Inf, `false` otherwise
"""
function nested_has_inf(ps)
    has_inf = Ref(false)
    fmap(ps) do x
        if x isa AbstractArray
            if any(isinf, x)
                has_inf[] = true
            end
        end
        return x
    end
    return has_inf[]
end

"""
    nested_all_zero(ps)

Check if all arrays in a nested parameter structure are zero.

# Arguments
- `ps`: Nested parameter structure (typically from Lux.jl)

# Returns
- `true` if all arrays are zero, `false` otherwise
"""
function nested_all_zero(ps)
    all_zero = Ref(true)
    fmap(ps) do x
        if x isa AbstractArray
            if !all(iszero, x)
                all_zero[] = false
            end
        end
        return x
    end
    return all_zero[]
end
