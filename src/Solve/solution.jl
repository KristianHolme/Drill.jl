struct RLSolution{U, P, A, ST, TS, B, TO}
    u::U
    prob::P
    alg::A
    retcode::ReturnCode.T
    stats::ST
    train_state::TS
    buffer::B
    timer::TO
end

function RLSolution(cache::RLCache)
    u = (parameters(cache), states(cache))
    return RLSolution(
        u,
        cache.prob,
        cache.alg,
        cache.retcode,
        cache.stats,
        cache.train_state,
        cache.buffer,
        cache.timer,
    )
end
