# Space utilities that depend on OneHotArrays
# These are kept in Drill.jl (not DrillInterface) because they require OneHotArrays.jl

using OneHotArrays
using DrillInterface: Discrete

"""
    discrete_to_onehotbatch(actions::AbstractArray{<:Integer}, space::Discrete)

Convert an array of discrete actions to a one-hot batch matrix.
"""
function discrete_to_onehotbatch(actions::AbstractArray{<:Integer}, space::Discrete)
    flat_actions = vec(actions)
    indices = map(flat_actions) do action
        idx = action - space.start + 1
        @assert 1 <= idx <= space.n "Action $(action) is out of bounds for Discrete($(space.n), $(space.start))"
        return idx
    end
    return onehotbatch(indices, 1:space.n)
end

"""
    onehotbatch_to_discrete(actions::AbstractMatrix, space::Discrete)

Convert a one-hot batch matrix back to discrete actions.
"""
function onehotbatch_to_discrete(actions::AbstractMatrix, space::Discrete)
    idx = argmax(actions; dims = 1)
    return [space.start + idx[i][1] - 1 for i in eachindex(idx)]
end

export discrete_to_onehotbatch, onehotbatch_to_discrete
