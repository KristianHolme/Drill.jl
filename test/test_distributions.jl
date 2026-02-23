@testitem "DiagGaussian vs Distributions.MvNormal" begin
    using Random
    using Distributions
    using LinearAlgebra
    using Drill.DrillDistributions


    shapes = [(1,), (1, 1), (2,), (2, 3), (2, 3, 1), (2, 3, 4)]
    for shape in shapes
        low = rand(Float32, shape...) .- 1.0f0
        high = rand(Float32, shape...) .+ 1.0f0
        action_space = Box(low, high, shape)

        same_outputs = Bool[]
        for i in 1:100
            mean = rand(action_space)
            log_std = rand(action_space)

            flat_mean = vec(mean)
            flat_log_std = vec(log_std)
            mvn = MvNormal(flat_mean, LinearAlgebra.Diagonal(map(abs2, exp.(flat_log_std))))

            d = DiagGaussian(mean, log_std)
            x = rand(action_space)

            flat_x = vec(x)

            custom_logpdf = DrillDistributions.logpdf(d, x)
            dist_logpdf = Distributions.logpdf(mvn, flat_x)
            push!(same_outputs, custom_logpdf ≈ dist_logpdf)

            custom_entropy = DrillDistributions.entropy(d)
            dist_entropy = Distributions.entropy(mvn)
            push!(same_outputs, custom_entropy ≈ dist_entropy)
        end
        @test all(same_outputs)
    end

end

@testitem "SquashedDiagGaussian epsilon keyword types" begin
    using Random
    using Drill.DrillDistributions

    make_triplet(::Type{T}) where {T <: AbstractFloat} = (rand(T, 2, 3), rand(T, 2, 3), rand(T, 2, 3))

    function check_ok_with_eps(::Type{T}) where {T <: AbstractFloat}
        mean, log_std, x = make_triplet(T)
        d = SquashedDiagGaussian(mean, log_std, T(1.0e-6))
        y = DrillDistributions.logpdf(d, x)
        @test y isa T
        @test d.epsilon isa T
        true
    end

    # matching types should succeed and preserve type
    @test check_ok_with_eps(Float32)
    @test check_ok_with_eps(Float64)

    # mismatched epsilon type should fail
    mean32, logstd32, _ = make_triplet(Float32)
    mean64, logstd64, _ = make_triplet(Float64)
    @test_throws MethodError SquashedDiagGaussian(mean32, logstd32, Float64(1.0e-6))
    @test_throws MethodError SquashedDiagGaussian(mean64, logstd64, Float32(1.0e-6))
    @test_throws MethodError SquashedDiagGaussian(mean64, logstd64, Float16(1.0e-6))
end


@testitem "SquashedDiagGaussian constructor and logpdf types" begin
    using Random
    using Drill.DrillDistributions

    make_triplet(::Type{T}) where {T <: AbstractFloat} = (rand(T, 2, 3), rand(T, 2, 3), rand(T, 2, 3))

    function check_ok_and_type(::Type{T}) where {T <: AbstractFloat}
        mean, log_std, x = make_triplet(T)
        d = SquashedDiagGaussian(mean, log_std)
        y = DrillDistributions.logpdf(d, x)
        @test y isa T
        true
    end

    # same-type constructions should work and logpdf should return matching type
    @test check_ok_and_type(Float32)
    @test check_ok_and_type(Float64)

    # mixed-type constructions should fail
    mean32, logstd32, _ = make_triplet(Float32)
    mean64, logstd64, _ = make_triplet(Float64)
    @test_throws MethodError SquashedDiagGaussian(mean32, logstd64)
    @test_throws MethodError SquashedDiagGaussian(mean64, logstd32)
end

@testitem "Categorical vs Distributions.Categorical" begin
    using Random
    using Distributions

    same_outputs = Bool[]

    for N in [3, 8], i in 1:100
        p = rand(Float32, N)
        p = p ./ sum(p)

        d = DrillDistributions.BatchedCategorical()

        dist_d = Distributions.Categorical(p)

        probs = reshape(p, :, 1)
        x = zeros(Float32, N, 1)
        x[1, 1] = 1.0f0
        custom_logpdf = DrillDistributions.logpdf(d, x, probs)[1]
        dist_logpdf = Distributions.logpdf(dist_d, 1)
        push!(same_outputs, custom_logpdf ≈ dist_logpdf)

        custom_entropy = DrillDistributions.entropy(d, probs)[1]
        dist_entropy = Distributions.entropy(dist_d)
        push!(same_outputs, custom_entropy ≈ dist_entropy)
    end

    @test all(same_outputs)
end


@testitem "Diaggaussion constructor" begin
    using Random

    same_outputs = Bool[]

    #test strict types
    @test_throws MethodError DiagGaussian([1.0], [2.0f0])

    d = DiagGaussian([1.0f0], [2.0f0])
    @test_throws MethodError DrillDistributions.logpdf(d, [1.0])


    mean_batch = rand(Float32, 2, 2, 7)
    std_batch = rand(Float32, 2, 2, 7)

    x_batch = rand(Float32, 2, 2, 7)

    @test begin
        ds = DiagGaussian.(eachslice(mean_batch, dims = ndims(mean_batch)), eachslice(std_batch, dims = ndims(std_batch)))
        entropies = DrillDistributions.entropy.(ds)
        logpdfs = DrillDistributions.logpdf.(ds, eachslice(x_batch, dims = ndims(x_batch)))
        true
    end

    single_std = rand(Float32, 2, 2)
    @test begin
        ds = DiagGaussian.(eachslice(mean_batch, dims = ndims(mean_batch)), Ref(single_std))
        entropies = DrillDistributions.entropy.(ds)
        logpdfs = DrillDistributions.logpdf.(ds, eachslice(x_batch, dims = ndims(x_batch)))
        true
    end
end
