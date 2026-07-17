struct RLProblem{E, M, U, A}
    env::E
    model::M
    u0::U
    adapter::A
end

function RLProblem(env, model; u0 = nothing, adapter = nothing)
    return RLProblem{typeof(env), typeof(model), typeof(u0), typeof(adapter)}(
        env,
        model,
        u0,
        adapter,
    )
end
