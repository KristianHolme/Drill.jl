# Benchmark: old BatchedCategorical rand (loop + onehotbatch) vs new (Gumbel-max tensor).
# Optional: Reactant compiled kernel, Profile section. Run each ## section in order or as needed.
# Load with: includet("_research/performance/categorical_rand_bench.jl")

using BenchmarkTools
using OneHotArrays
using Random

## Implementations (old vs new)
"""Old implementation: loop over batch, Vector{Int}, OneHotArrays.onehotbatch."""
function old_rand(rng::AbstractRNG, probs::AbstractMatrix)
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

"""New implementation: Gumbel-max, tensor ops only, dense matrix output."""
function new_rand(rng::AbstractRNG, probs::AbstractMatrix{T}) where {T}
    U = rand(rng, T, size(probs))
    U = clamp.(U, eps(T), 1 - eps(T))
    G = -log.(.-log.(U))
    log_probs = log.(max.(probs, eps(T)))
    scores = log_probs .+ G
    one_hot = (scores .== maximum(scores, dims = 1))
    return T.(one_hot)
end

"""Optimized Drill implementation: Gumbel-max with findmax + one-hot."""
function drill_rand(rng::AbstractRNG, probs::AbstractMatrix{T}) where {T}
    # Optimized Gumbel-max trick for batched sampling
    epsval = eps(T)
    clamped_probs = clamp.(probs, epsval, one(T)-epsval)
    
    # Generate Gumbel noise
    U = rand(rng, T, size(probs))
    U = clamp.(U, epsval, one(T)-epsval)
    G = -log.(-log.(U))
    
    # Compute scores
    log_probs = log.(clamped_probs)
    scores = log_probs .+ G
    
    # Find max along class dimension (rows) for each batch element
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

## Benchmarks: old vs new (CPU)
rng = MersenneTwister(12345)
println("=== BatchedCategorical rand: old (loop+onehotbatch) vs new (Gumbel-max tensor) ===\n")
println("  Also comparing with Drill optimized implementation\n")
for (num_classes, batch_size) in [(4, 32), (16, 256), (64, 1024), (256, 4096)]
    probs = rand(rng, Float32, num_classes, batch_size)
    probs = probs ./ sum(probs, dims = 1)
    println("Shape ($num_classes × $batch_size):")
    old_elapsed = @elapsed for _ in 1:100
        old_rand(rng, probs)
    end
    new_elapsed = @elapsed for _ in 1:100
        new_rand(rng, probs)
    end
    drill_elapsed = @elapsed for _ in 1:100
        drill_rand(rng, probs)
    end
    println("  (100 runs) old $(round(old_elapsed * 1000, digits = 2)) ms  new $(round(new_elapsed * 1000, digits = 2)) ms  drill $(round(drill_elapsed * 1000, digits = 2)) ms")
    println("           ratio new/old: $(round(new_elapsed / old_elapsed, digits = 2))x  drill/old: $(round(drill_elapsed / old_elapsed, digits = 2))x")
end
println("\n--- @benchmark single call ---")
probs = rand(MersenneTwister(1), Float32, 16, 256)
probs = probs ./ sum(probs, dims = 1)
rng_old = MersenneTwister(2)
rng_new = MersenneTwister(2)
rng_drill = MersenneTwister(2)
@info "old:"
display(@benchmark old_rand($rng_old, $probs))
@info "new:"
display(@benchmark new_rand($rng_new, $probs))
@info "drill:"
display(@benchmark drill_rand($rng_drill, $probs))

## Profile: run many times under Profile.@profile then open @profview
n = 50_000
num_classes, batch_size = 16, 256
rng_profile = Random.Xoshiro(42)
probs_profile = rand(rng_profile, Float32, num_classes, batch_size)
probs_profile = probs_profile ./ sum(probs_profile, dims = 1)
rng_inner = Random.Xoshiro(43)
@profview for _ in 1:n
    new_rand(rng_inner, probs_profile)
end


## Profile (Drill path): run after loading Drill; profiles categorical_rand_kernel
# using Drill
# function categorical_rand_kernel(rng, probs)
#     d = Drill.DrillDistributions.BatchedCategorical()
#     return rand(rng, d, probs)
# end
# Profile.@profile for _ in 1:50_000
#     categorical_rand_kernel(rng_inner, probs_profile)
# end
# ProfileView.@profview

## Reactant: CPU new vs compiled (run after: using Drill, Reactant, Lux, Adapt)
using Drill
using Reactant
using Lux
using Adapt
function categorical_rand_kernel(rng, probs)
    d = Drill.DrillDistributions.BatchedCategorical()
    return rand(rng, d, probs)
end
Reactant.set_default_backend("cpu")
device = Lux.reactant_device()
println("=== BatchedCategorical rand: CPU new vs Reactant compiled (Gumbel-max) ===\n")
println("(Old implementation cannot be compiled with Reactant: uses Vector{Int}, loops, OneHotArrays.)\n")
for (num_classes, batch_size) in [(16, 256), (64, 1024)]
    probs_cpu = rand(Random.Xoshiro(1), Float32, num_classes, batch_size)
    probs_cpu = probs_cpu ./ sum(probs_cpu, dims = 1)
    probs_dev = Adapt.adapt(device, probs_cpu)
    rng_dev = Adapt.adapt(device, Random.Xoshiro(2))
    println("Shape ($num_classes × $batch_size):")
    print("  CPU new (100 runs): ")
    rng_cpu = Random.Xoshiro(3)
    cpu_elapsed = @elapsed for _ in 1:100
        categorical_rand_kernel(rng_cpu, probs_cpu)
    end
    println("$(round(cpu_elapsed * 1000, digits = 2)) ms")
    print("  Compiling Reactant kernel... ")
    compiled = try
        Reactant.@compile categorical_rand_kernel(rng_dev, probs_dev)
    catch e
        @warn "Reactant @compile failed" exception = e
        println("FAILED")
        continue
    end
    println("ok.")
    for _ in 1:3
        rng_warm = Adapt.adapt(device, Random.Xoshiro(rand(UInt64)))
        compiled(rng_warm, probs_dev)
    end
    print("  Reactant compiled (100 runs): ")
    rng_100 = Adapt.adapt(device, Random.Xoshiro(4))
    react_elapsed = @elapsed for _ in 1:100
        compiled(rng_100, probs_dev)
    end
    println("$(round(react_elapsed * 1000, digits = 2)) ms  (vs CPU: $(round(react_elapsed / cpu_elapsed, digits = 2))x)")
end
println("\n--- @benchmark single call (16×256) ---")
probs_cpu = rand(Random.Xoshiro(10), Float32, 16, 256)
probs_cpu = probs_cpu ./ sum(probs_cpu, dims = 1)
probs_dev = Adapt.adapt(device, probs_cpu)
rng_cpu = Random.Xoshiro(11)
rng_dev = Adapt.adapt(device, Random.Xoshiro(12))
compiled = Reactant.@compile categorical_rand_kernel(rng_dev, probs_dev)
println("CPU:")
display(@benchmark categorical_rand_kernel($rng_cpu, $probs_cpu))
rng_dev_bench = Adapt.adapt(device, Random.Xoshiro(13))
println("Reactant compiled:")
display(@benchmark $compiled($rng_dev_bench, $probs_dev))
