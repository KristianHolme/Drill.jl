import ..Solve: RLCache, training_progress

"""
    update_learning_rate!(cache, alg) -> lr

Evaluate the learning rate of `alg` at the current training progress and, when it is a
schedule, set it in the optimizer states of `cache.learner`. Returns the learning rate.
"""
function update_learning_rate!(cache::RLCache, alg::AbstractAlgorithm)
    lr = learning_rate(alg, training_progress(cache))
    is_constant(alg.learning_rate) || adjust_learning_rate!(cache.learner, lr)
    return lr
end
