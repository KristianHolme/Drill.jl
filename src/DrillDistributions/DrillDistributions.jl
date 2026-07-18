module DrillDistributions

#TODO: remove lux dependency, find other source of softplus
using Lux: softplus
import Random: AbstractRNG, rand, randn

export DiagGaussian, SquashedDiagGaussian
export BatchedCategorical, BatchedDiagGaussian, BatchedSquashedDiagGaussian

export logpdf, entropy, mode


abstract type AbstractDistribution end

abstract type AbstractContinuousDistribution <: AbstractDistribution end

abstract type AbstractDiscreteDistribution <: AbstractDistribution end

include("categorical.jl")
include("diagGaussian.jl")
include("squashedDiagGaussian.jl")

end
