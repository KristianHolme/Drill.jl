using Test
using Drill
using DrillInterface: Box, Discrete
using Random

@testset "Box space creation and properties" begin
    low = Float32[-2.0, -1.0]
    high = Float32[1.0, 3.0]
    space = Box(low, high)

    @test typeof(space) == Box{Float32}
    @test space.low == low
    @test space.high == high
    @test space.shape == (2,)

    space_1d = Box(Float32[-5.0], Float32[10.0])
    @test space_1d.shape == (1,)
    @test space_1d.low == Float32[-5.0]
    @test space_1d.high == Float32[10.0]

    low_3d = Float32[-1.0, 0.0, -10.0]
    high_3d = Float32[1.0, 5.0, 0.0]
    space_3d = Box(low_3d, high_3d)
    @test space_3d.shape == (3,)
    @test space_3d.low == low_3d
    @test space_3d.high == high_3d
end

@testset "Box space validation" begin
    @test Box(Float32[-1.0, -2.0], Float32[1.0, 2.0]) isa Box{Float32}

    @test_throws AssertionError Box(Float32[-1.0], Float32[1.0, 2.0])

    @test_throws AssertionError Box(Float32[1.0, -1.0], Float32[0.0, 1.0])

    space_equal = Box(Float32[1.0, 2.0], Float32[1.0, 2.0])
    @test space_equal isa Box{Float32}
end

@testset "Random sampling from Box" begin
    low = Float32[-3.0, 0.0]
    high = Float32[2.0, 10.0]
    space = Box(low, high)
    rng = MersenneTwister(42)

    sample1 = rand(rng, space)
    @test length(sample1) == 2
    @test low[1] ≤ sample1[1] ≤ high[1]
    @test low[2] ≤ sample1[2] ≤ high[2]
    @test eltype(sample1) == Float32

    sample2 = rand(space)
    @test length(sample2) == 2
    @test low[1] ≤ sample2[1] ≤ high[1]
    @test low[2] ≤ sample2[2] ≤ high[2]
    @test eltype(sample2) == Float32

    n = 5
    samples = rand(rng, space, n)
    @test size(samples) == (n,)
    @test all(map(action -> all(low .≤ action .≤ high), samples))
    @test eltype(samples) == Vector{Float32}

    samples2 = rand(space, n)
    @test size(samples2) == (n,)
    @test size(samples2[1]) == (2,)
    @test eltype(samples2) == Vector{Float32}
    @test all(map(action -> eltype(action) == Float32, samples2))
end

@testset "Box containment checking" begin
    low = Float32[-2.0, 1.0]
    high = Float32[0.0, 5.0]
    space = Box(low, high)

    valid_samples = [
        Float32[-1.0, 3.0],
        Float32[-2.0, 1.0],
        Float32[0.0, 5.0],
        Float32[-1.5, 2.5],
    ]

    for sample in valid_samples
        @test sample ∈ space
        @test sample in space
    end

    invalid_samples = [
        Float32[0.5, 3.0],
        Float32[-1.0, 0.5],
        Float32[-3.0, 6.0],
        Float32[-1.0, 3.0, 0.0],
    ]

    for sample in invalid_samples
        @test !(sample ∈ space)
        @test !(sample in space)
    end

    rng = MersenneTwister(123)
    for i in 1:10
        sample = rand(rng, space)
        @test sample ∈ space
    end
end

@testset "Box edge cases and special configurations" begin
    tiny_space = Box(Float32[0.0], Float32[1.0e-6])
    @test Float32[0.0] ∈ tiny_space
    @test Float32[1.0e-6] ∈ tiny_space
    @test !(Float32[1.0e-5] ∈ tiny_space)

    neg_space = Box(Float32[-10.0, -5.0], Float32[-1.0, -2.0])
    @test Float32[-5.0, -3.0] ∈ neg_space
    @test !(Float32[0.0, -3.0] ∈ neg_space)

    mixed_space = Box(Float32[-1.0, 2.0], Float32[1.0, 8.0])
    @test Float32[0.0, 5.0] ∈ mixed_space
    @test !(Float32[2.0, 1.0] ∈ mixed_space)

    point_space = Box(Float32[1.0, 2.0], Float32[1.0, 2.0])
    @test Float32[1.0, 2.0] ∈ point_space
    @test !(Float32[1.0, 2.1] ∈ point_space)
end

@testset "Box interface completeness" begin
    space = Box(Float32[-1.0, 0.0], Float32[1.0, 5.0])

    @test hasmethod(rand, (AbstractRNG, typeof(space)))
    @test hasmethod(rand, (typeof(space),))
    @test hasmethod(rand, (AbstractRNG, typeof(space), Int))
    @test hasmethod(rand, (typeof(space), Int))
    @test hasmethod(in, (Vector{Float32}, typeof(space)))

    rng = MersenneTwister(42)

    s1 = rand(rng, space)
    s2 = rand(space)
    @test s1 ∈ space
    @test s2 ∈ space

    s3 = rand(rng, space, 3)
    s4 = rand(space, 3)
    @test size(s3) == (3,)
    @test size(s4) == (3,)
    @test all(map(s -> s ∈ space, s3))
    @test all(map(s -> s ∈ space, s4))
end

@testset "Discrete space creation and properties" begin
    space_default = Discrete(5)
    @test space_default.n == 5
    @test space_default.start == 1
    @test eltype(space_default) == Int
    @test ndims(space_default) == 1

    space_custom = Discrete(3, 1)
    @test space_custom.n == 3
    @test space_custom.start == 1

    space_negative = Discrete(4, -2)
    @test space_negative.n == 4
    @test space_negative.start == -2

    space1 = Discrete(5, 0)
    space2 = Discrete(5, 0)
    space3 = Discrete(5, 1)
    space4 = Discrete(4, 0)

    @test isequal(space1, space2)
    @test !isequal(space1, space3)
    @test !isequal(space1, space4)
end

@testset "Random sampling from Discrete" begin
    space_0 = Discrete(5, 0)
    rng = MersenneTwister(42)

    sample1 = rand(rng, space_0)
    @test sample1 isa Int
    @test 0 ≤ sample1 ≤ 4
    @test sample1 ∈ space_0

    sample2 = rand(space_0)
    @test sample2 isa Int
    @test 0 ≤ sample2 ≤ 4
    @test sample2 ∈ space_0

    n = 20
    samples = rand(rng, space_0, n)
    @test length(samples) == n
    @test all(s -> s isa Int, samples)
    @test all(s -> 0 ≤ s ≤ 4, samples)
    @test all(s -> s ∈ space_0, samples)

    samples2 = rand(space_0, n)
    @test length(samples2) == n
    @test all(s -> s isa Int, samples2)
    @test all(s -> 0 ≤ s ≤ 4, samples2)

    space_1 = Discrete(3, 1)
    samples_1based = rand(rng, space_1, 15)
    @test all(s -> 1 ≤ s ≤ 3, samples_1based)
    @test all(s -> s ∈ space_1, samples_1based)

    space_custom = Discrete(4, -1)
    samples_custom = rand(rng, space_custom, 15)
    @test all(s -> -1 ≤ s ≤ 2, samples_custom)
    @test all(s -> s ∈ space_custom, samples_custom)
end

@testset "Discrete containment checking" begin
    space_0 = Discrete(5, 0)

    valid_values = [0, 1, 2, 3, 4]
    @test all(val -> val ∈ space_0, valid_values)
    @test all(val -> val in space_0, valid_values)

    invalid_values = [-1, 5, 6, 10]
    @test all(val -> !(val ∈ space_0), invalid_values)
    @test all(val -> !(val in space_0), invalid_values)

    space_1 = Discrete(3, 1)
    @test all(val -> val ∈ space_1 && val in space_1, [1, 2, 3])
    @test all(val -> !(val ∈ space_1 && val in space_1), [0, 4])

    space_custom = Discrete(4, -2)
    @test all(val -> val ∈ space_custom && val in space_custom, [-2, -1, 0, 1])
    @test all(val -> !(val ∈ space_custom && val in space_custom), [-3, 2])

    @test !(1.0 ∈ space_0)
    @test !(1.5 ∈ space_0)
    @test !("1" ∈ space_0)
    @test !([1] ∈ space_0)
end

@testset "Discrete action processing" begin
    using Drill: process_action
    space_0 = Discrete(5, 0)
    alg = PPO()

    @test all(process_action(0:4, space_0, alg) .== 0:4)

    @test process_action(0, space_0, alg) == 0
    @test_throws AssertionError process_action(10, space_0, alg)
    @test_throws AssertionError process_action(-1, space_0, alg)
    @test_throws MethodError process_action(1.0, space_0, alg)
    @test_throws MethodError process_action(1.0f0, space_0, alg)

    space_1 = Discrete(3, 1)

    @test process_action(1, space_1, alg) == 1
    @test_throws AssertionError process_action(4, space_1, alg)
    @test_throws AssertionError process_action(0, space_1, alg)

    space_custom = Discrete(4, -1)
    @test_throws AssertionError process_action(5, space_custom, alg)
    @test_throws AssertionError process_action(-2, space_custom, alg)
    @test_throws MethodError process_action(1.0, space_custom, alg)
    @test_throws MethodError process_action(1.0f0, space_custom, alg)
end

@testset "Discrete space interface completeness" begin
    space = Discrete(4, 0)

    @test hasmethod(rand, (AbstractRNG, typeof(space)))
    @test hasmethod(rand, (typeof(space),))
    @test hasmethod(rand, (AbstractRNG, typeof(space), Int))
    @test hasmethod(rand, (typeof(space), Int))
    @test hasmethod(in, (Int, typeof(space)))
    @test hasmethod(Base.size, (typeof(space),))
    @test hasmethod(Base.ndims, (typeof(space),))
    @test hasmethod(Base.eltype, (typeof(space),))

    rng = MersenneTwister(42)

    s1 = rand(rng, space)
    s2 = rand(space)
    @test s1 ∈ space
    @test s2 ∈ space

    s3 = rand(rng, space, 5)
    s4 = rand(space, 5)
    @test length(s3) == 5
    @test length(s4) == 5
    @test all(s -> s ∈ space, s3)
    @test all(s -> s ∈ space, s4)

    @test size(space) == (1,)
    @test ndims(space) == 1
    @test eltype(space) == Int
end

@testset "Discrete space edge cases" begin
    using Drill: process_action
    alg = PPO()

    space_single = Discrete(1, 0)
    @test space_single.n == 1
    @test space_single.start == 0
    @test 0 ∈ space_single
    @test !(1 ∈ space_single)
    @test !(-1 ∈ space_single)

    @test_throws AssertionError Discrete(-1)
    @test_throws AssertionError Discrete(0)

    @test process_action(0, space_single, alg) == 0
    @test_throws AssertionError process_action(1, space_single, alg)
    @test_throws AssertionError process_action(-1, space_single, alg)

    space_single_1 = Discrete(1, 5)
    @test 5 ∈ space_single_1
    @test !(4 ∈ space_single_1)
    @test !(6 ∈ space_single_1)
    @test_throws AssertionError process_action(1, space_single_1, alg) == 5

    space_large = Discrete(1000, 0)
    @test 0 ∈ space_large
    @test 999 ∈ space_large
    @test !(1000 ∈ space_large)
    @test !(-1 ∈ space_large)

    samples = rand(space_large, 100)
    @test minimum(samples) >= 0
    @test maximum(samples) <= 999
    @test length(unique(samples)) > 50

    space_neg = Discrete(5, -10)
    @test (-10) ∈ space_neg
    @test (-6) ∈ space_neg
    @test !((-11) ∈ space_neg)
    @test !((-5) ∈ space_neg)

    @test_throws AssertionError process_action(1, space_neg, alg)
    @test_throws AssertionError process_action(5, space_neg, alg)
    @test_throws AssertionError process_action(0, space_neg, alg)
    @test_throws AssertionError process_action(6, space_neg, alg)
end
