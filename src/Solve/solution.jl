struct RLSolution{U, P, A, ST, LS, B, TO}
    u::U
    prob::P
    alg::A
    retcode::ReturnCode.T
    stats::ST
    learner::LS
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
        cache.learner,
        cache.buffer,
        cache.timer,
    )
end
