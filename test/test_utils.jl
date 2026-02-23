using Test
using Drill

@testitem "utils.jl" begin
    using ComponentArrays
    target = [1.0, 2.0, 3.0]
    source = [0.0, 0.0, 0.0]
    tau = 0.5
    Drill.polyak_update!(target, source, tau)
    @test target == [0.5, 1.0, 1.5]

    target = ComponentArray(a = [1.0], b = [2.0], c = [3.0])
    source = ComponentArray(a = [0.0], b = [0.0], c = [0.0], d = [0.0])
    tau = 0.5
    Drill.polyak_update!(target, source, tau)
    @test target.a[1] == 0.5
    @test target.b[1] == 1.0
    @test target.c[1] == 1.5

    target = [0.0, 0.0, 0.0]
    source = [1.0, 2.0, 3.0]
    tau = 0.01
    Drill.polyak_update!(target, source, tau)
    @test target == [0.01, 0.02, 0.03]
end
