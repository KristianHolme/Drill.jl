# Tests for the general Box space with different bounds per dimension
@testitem "Box space creation and properties" tags = [:spaces, :basic, :box] begin
    using Random

    # Test basic Box creation with different bounds per dimension
    low = Float32[-2.0, -1.0]
    high = Float32[1.0, 3.0]
    space = Box(low, high)

    @test typeof(space) == Box{Float32}
    @test space.low == low
    @test space.high == high
    @test space.shape == (2,)

    # Test single dimension Box
    space_1d = Box(Float32[-5.0], Float32[10.0])
    @test space_1d.shape == (1,)
    @test space_1d.low == Float32[-5.0]
    @test space_1d.high == Float32[10.0]

    # Test higher dimensional Box with varied bounds
    low_3d = Float32[-1.0, 0.0, -10.0]
    high_3d = Float32[1.0, 5.0, 0.0]
    space_3d = Box(low_3d, high_3d)
    @test space_3d.shape == (3,)
    @test space_3d.low == low_3d
    @test space_3d.high == high_3d
end

@testitem "Box space validation" tags = [:spaces, :validation, :box] begin
    # Test Box constructor validation

    # Should work - valid bounds
    @test Box(Float32[-1.0, -2.0], Float32[1.0, 2.0]) isa Box{Float32}

    # Should fail - mismatched shapes
    @test_throws AssertionError Box(Float32[-1.0], Float32[1.0, 2.0])

    # Should fail - low > high in some dimension
    @test_throws AssertionError Box(Float32[1.0, -1.0], Float32[0.0, 1.0])

    # Edge case - low == high (valid)
    space_equal = Box(Float32[1.0, 2.0], Float32[1.0, 2.0])
    @test space_equal isa Box{Float32}
end

@testitem "Random sampling from Box" tags = [:spaces, :random, :box] begin
    using Random

    # Test Box with different bounds per dimension
    low = Float32[-3.0, 0.0]
    high = Float32[2.0, 10.0]
    space = Box(low, high)
    rng = MersenneTwister(42)

    # Test rand(rng, space)
    sample1 = rand(rng, space)
    @test length(sample1) == 2
    @test low[1] ≤ sample1[1] ≤ high[1]
    @test low[2] ≤ sample1[2] ≤ high[2]
    @test eltype(sample1) == Float32

    # Test rand(space) - default RNG
    sample2 = rand(space)
    @test length(sample2) == 2
    @test low[1] ≤ sample2[1] ≤ high[1]
    @test low[2] ≤ sample2[2] ≤ high[2]
    @test eltype(sample2) == Float32

    # Test rand(rng, space, n) - multiple samples
    n = 5
    samples = rand(rng, space, n)
    @test size(samples) == (n,)
    @test all(map(action -> all(low .≤ action .≤ high), samples))
    # for i in 1:n
    #     @test low[1] ≤ samples[1, i] ≤ high[1]
    #     @test low[2] ≤ samples[2, i] ≤ high[2]
    # end
    @test eltype(samples) == Vector{Float32}

    # Test rand(space, n) - multiple samples with default RNG
    samples2 = rand(space, n)
    @test size(samples2) == (n,)
    @test size(samples2[1]) == (2,)
    @test eltype(samples2) == Vector{Float32}
    @test all(map(action -> eltype(action) == Float32, samples2))
end

@testitem "Box containment checking" tags = [:spaces, :containment, :box] begin
    using Random

    # Test Box with asymmetric bounds
    low = Float32[-2.0, 1.0]
    high = Float32[0.0, 5.0]
    space = Box(low, high)

    # Test valid samples are in space
    valid_samples = [
        Float32[-1.0, 3.0],
        Float32[-2.0, 1.0],  # Lower boundary
        Float32[0.0, 5.0],   # Upper boundary
        Float32[-1.5, 2.5],
    ]

    for sample in valid_samples
        @test sample ∈ space
        @test sample in space
    end

    # Test invalid samples are not in space
    invalid_samples = [
        Float32[0.5, 3.0],    # First component too high
        Float32[-1.0, 0.5],   # Second component too low
        Float32[-3.0, 6.0],   # Both components out of bounds
        Float32[-1.0, 3.0, 0.0],  # Wrong dimensions
    ]

    for sample in invalid_samples
        @test !(sample ∈ space)
        @test !(sample in space)
    end

    # Test with generated samples
    rng = MersenneTwister(123)
    for i in 1:10
        sample = rand(rng, space)
        @test sample ∈ space
    end
end

@testitem "Box edge cases and special configurations" tags = [:spaces, :box, :edge_cases] begin
    # Test very small bounds differences
    tiny_space = Box(Float32[0.0], Float32[1.0e-6])
    @test Float32[0.0] ∈ tiny_space
    @test Float32[1.0e-6] ∈ tiny_space
    @test !(Float32[1.0e-5] ∈ tiny_space)

    # Test negative bounds
    neg_space = Box(Float32[-10.0, -5.0], Float32[-1.0, -2.0])
    @test Float32[-5.0, -3.0] ∈ neg_space
    @test !(Float32[0.0, -3.0] ∈ neg_space)

    # Test mixed positive/negative bounds
    mixed_space = Box(Float32[-1.0, 2.0], Float32[1.0, 8.0])
    @test Float32[0.0, 5.0] ∈ mixed_space
    @test !(Float32[2.0, 1.0] ∈ mixed_space)

    # Test single point space (low == high)
    point_space = Box(Float32[1.0, 2.0], Float32[1.0, 2.0])
    @test Float32[1.0, 2.0] ∈ point_space
    @test !(Float32[1.0, 2.1] ∈ point_space)
end

@testitem "Box interface completeness" tags = [:spaces, :interface, :box] begin
    using Random

    # Test that Box implements all expected interface methods
    space = Box(Float32[-1.0, 0.0], Float32[1.0, 5.0])

    # Test that all expected methods exist
    @test hasmethod(rand, (AbstractRNG, typeof(space)))
    @test hasmethod(rand, (typeof(space),))
    @test hasmethod(rand, (AbstractRNG, typeof(space), Int))
    @test hasmethod(rand, (typeof(space), Int))
    @test hasmethod(in, (Vector{Float32}, typeof(space)))

    # Test that methods work as expected
    rng = MersenneTwister(42)

    # Single sample methods
    s1 = rand(rng, space)
    s2 = rand(space)
    @test s1 ∈ space
    @test s2 ∈ space

    # Multiple sample methods
    s3 = rand(rng, space, 3)
    s4 = rand(space, 3)
    @test size(s3) == (3,)
    @test size(s4) == (3,)
    @test all(map(s -> s ∈ space, s3))
    @test all(map(s -> s ∈ space, s4))
end

# Tests for Discrete action spaces
@testitem "Discrete space creation and properties" tags = [:spaces, :basic, :discrete] begin
    # Test basic Discrete creation with default start (0-based)
    space_default = Discrete(5)
    @test space_default.n == 5
    @test space_default.start == 1
    @test eltype(space_default) == Int
    @test ndims(space_default) == 1

    # Test Discrete creation with custom start (1-based)
    space_custom = Discrete(3, 1)
    @test space_custom.n == 3
    @test space_custom.start == 1

    # Test Discrete creation with negative start
    space_negative = Discrete(4, -2)
    @test space_negative.n == 4
    @test space_negative.start == -2

    # Test equality
    space1 = Discrete(5, 0)
    space2 = Discrete(5, 0)
    space3 = Discrete(5, 1)
    space4 = Discrete(4, 0)

    @test isequal(space1, space2)
    @test !isequal(space1, space3)  # Different start
    @test !isequal(space1, space4)  # Different n
end

@testitem "Random sampling from Discrete" tags = [:spaces, :random, :discrete] begin
    using Random

    # Test 0-based discrete space (default)
    space_0 = Discrete(5, 0)  # Values: 0, 1, 2, 3, 4
    rng = MersenneTwister(42)

    # Test rand(rng, space)
    sample1 = rand(rng, space_0)
    @test sample1 isa Int
    @test 0 ≤ sample1 ≤ 4
    @test sample1 ∈ space_0

    # Test rand(space) - default RNG
    sample2 = rand(space_0)
    @test sample2 isa Int
    @test 0 ≤ sample2 ≤ 4
    @test sample2 ∈ space_0

    # Test rand(rng, space, n) - multiple samples
    n = 20
    samples = rand(rng, space_0, n)
    @test length(samples) == n
    @test all(s -> s isa Int, samples)
    @test all(s -> 0 ≤ s ≤ 4, samples)
    @test all(s -> s ∈ space_0, samples)

    # Test rand(space, n) - multiple samples with default RNG
    samples2 = rand(space_0, n)
    @test length(samples2) == n
    @test all(s -> s isa Int, samples2)
    @test all(s -> 0 ≤ s ≤ 4, samples2)

    # Test 1-based discrete space
    space_1 = Discrete(3, 1)  # Values: 1, 2, 3
    samples_1based = rand(rng, space_1, 15)
    @test all(s -> 1 ≤ s ≤ 3, samples_1based)
    @test all(s -> s ∈ space_1, samples_1based)

    # Test custom start space
    space_custom = Discrete(4, -1)  # Values: -1, 0, 1, 2
    samples_custom = rand(rng, space_custom, 15)
    @test all(s -> -1 ≤ s ≤ 2, samples_custom)
    @test all(s -> s ∈ space_custom, samples_custom)
end

@testitem "Discrete containment checking" tags = [:spaces, :containment, :discrete] begin
    # Test 0-based discrete space
    space_0 = Discrete(5, 0)  # Values: 0, 1, 2, 3, 4

    # Test valid values
    valid_values = [0, 1, 2, 3, 4]
    @test all(val -> val ∈ space_0, valid_values)
    @test all(val -> val in space_0, valid_values)

    # Test invalid values
    invalid_values = [-1, 5, 6, 10]
    @test all(val -> !(val ∈ space_0), invalid_values)
    @test all(val -> !(val in space_0), invalid_values)

    # Test 1-based discrete space
    space_1 = Discrete(3, 1)  # Values: 1, 2, 3
    @test all(val -> val ∈ space_1 && val in space_1, [1, 2, 3])
    @test all(val -> !(val ∈ space_1 && val in space_1), [0, 4])

    # Test custom start space
    space_custom = Discrete(4, -2)  # Values: -2, -1, 0, 1
    @test all(val -> val ∈ space_custom && val in space_custom, [-2, -1, 0, 1])
    @test all(val -> !(val ∈ space_custom && val in space_custom), [-3, 2])

    # Test non-integer types are rejected
    @test !(1.0 ∈ space_0)
    @test !(1.5 ∈ space_0)
    @test !("1" ∈ space_0)
    @test !([1] ∈ space_0)
end

@testitem "Discrete action processing" tags = [:spaces, :discrete, :action_processing] begin
    using Drill: process_action
    # Test process_action for different discrete spaces

    # Test 0-based space (Gymnasium style)
    space_0 = Discrete(5, 0)  # Valid actions: 0, 1, 2, 3, 4
    alg = PPO()

    # Test basic conversion from 1-based to 0-based
    @test all(process_action(0:4, space_0, alg) .== 0:4)  # Julia 1-based → space 0-based

    # Test clamping for out-of-bounds actions
    @test process_action(0, space_0, alg) == 0    # Below valid range, clamp to min
    @test_throws AssertionError process_action(10, space_0, alg)   # above valid range
    @test_throws AssertionError process_action(-1, space_0, alg)   # below valid range
    @test_throws MethodError process_action(1.0, space_0, alg)   # not an integer
    @test_throws MethodError process_action(1.0f0, space_0, alg)   # not an integer

    space_1 = Discrete(3, 1)  # Valid actions: 1, 2, 3

    @test process_action(1, space_1, alg) == 1  # 1-based → 1-based (no change)
    @test_throws AssertionError process_action(4, space_1, alg)   # above valid range
    @test_throws AssertionError process_action(0, space_1, alg)   # below valid range

    # Test custom start space
    space_custom = Discrete(4, -1)  # Valid actions: -1, 0, 1, 2
    @test_throws AssertionError process_action(5, space_custom, alg)   # above valid range
    @test_throws AssertionError process_action(-2, space_custom, alg)   # below valid range
    @test_throws MethodError process_action(1.0, space_custom, alg)   # not an integer
    @test_throws MethodError process_action(1.0f0, space_custom, alg)   # not an integer
end


@testitem "Discrete space interface completeness" tags = [:spaces, :interface, :discrete] begin
    using Random

    # Test that Discrete implements all expected interface methods
    space = Discrete(4, 0)

    # Test that all expected methods exist
    @test hasmethod(rand, (AbstractRNG, typeof(space)))
    @test hasmethod(rand, (typeof(space),))
    @test hasmethod(rand, (AbstractRNG, typeof(space), Int))
    @test hasmethod(rand, (typeof(space), Int))
    @test hasmethod(in, (Int, typeof(space)))
    @test hasmethod(Base.size, (typeof(space),))
    @test hasmethod(Base.ndims, (typeof(space),))
    @test hasmethod(Base.eltype, (typeof(space),))

    # Test that methods work as expected
    rng = MersenneTwister(42)

    # Single sample methods
    s1 = rand(rng, space)
    s2 = rand(space)
    @test s1 ∈ space
    @test s2 ∈ space

    # Multiple sample methods
    s3 = rand(rng, space, 5)
    s4 = rand(space, 5)
    @test length(s3) == 5
    @test length(s4) == 5
    @test all(s -> s ∈ space, s3)
    @test all(s -> s ∈ space, s4)

    # Size and properties
    @test size(space) == (1,)
    @test ndims(space) == 1
    @test eltype(space) == Int
end

@testitem "Discrete space edge cases" tags = [:spaces, :discrete, :edge_cases] begin
    using Drill: process_action
    alg = PPO()

    # Test single action space
    space_single = Discrete(1, 0)  # Only action: 0
    @test space_single.n == 1
    @test space_single.start == 0
    @test 0 ∈ space_single
    @test !(1 ∈ space_single)
    @test !(-1 ∈ space_single)

    @test_throws AssertionError Discrete(-1)
    @test_throws AssertionError Discrete(0)

    # Test process_action with single action space
    @test process_action(0, space_single, alg) == 0  # Only valid action
    @test_throws AssertionError process_action(1, space_single, alg)  # out of bounds
    @test_throws AssertionError process_action(-1, space_single, alg)  # out of bounds

    # Test single action space with different start
    space_single_1 = Discrete(1, 5)  # Only action: 5
    @test 5 ∈ space_single_1
    @test !(4 ∈ space_single_1)
    @test !(6 ∈ space_single_1)
    @test_throws AssertionError process_action(1, space_single_1, alg) == 5

    # Test large action space
    space_large = Discrete(1000, 0)
    @test 0 ∈ space_large
    @test 999 ∈ space_large
    @test !(1000 ∈ space_large)
    @test !(-1 ∈ space_large)

    # Test random sampling from large space covers range
    samples = rand(space_large, 100)
    @test minimum(samples) >= 0
    @test maximum(samples) <= 999
    @test length(unique(samples)) > 50  # Should have good diversity

    # Test negative start with various boundaries
    space_neg = Discrete(5, -10)  # Actions: -10, -9, -8, -7, -6
    @test (-10) ∈ space_neg
    @test (-6) ∈ space_neg
    @test !((-11) ∈ space_neg)
    @test !((-5) ∈ space_neg)

    # Test process_action with negative start
    @test_throws AssertionError process_action(1, space_neg, alg)
    @test_throws AssertionError process_action(5, space_neg, alg)
    @test_throws AssertionError process_action(0, space_neg, alg)
    @test_throws AssertionError process_action(6, space_neg, alg)
end
