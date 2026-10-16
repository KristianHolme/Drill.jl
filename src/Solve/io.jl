function save_model_params_and_state(cache::RLCache, path::AbstractString; suffix::String = ".jld2")
    file_path = endswith(path, suffix) ? path : path * suffix
    host = cpu_device()
    save(
        file_path,
        Dict(
            "model" => cache.model,
            "parameters" => host(parameters(cache)),
            "states" => host(states(cache)),
            "learner" => host(cache.learner),
        ),
    )
    return file_path
end

function load_model_params_and_state!(
        cache::RLCache,
        alg::AbstractAlgorithm,
        path::AbstractString;
        suffix::String = ".jld2",
    )
    file_path = endswith(path, suffix) ? path : path * suffix
    dev = current_device(parameters(cache))
    data = load(file_path)
    model_key = haskey(data, "model") ? "model" : "layer"
    cache.model = data[model_key]
    learner = init_learner(
        alg, cache.model, dev(data["parameters"]), dev(data["states"]);
        rng = device_rng(dev, cache.rng), device = dev,
    )
    if haskey(data, "learner")
        saved = data["learner"]
        learner = dev isa CPUDevice ? saved : _copy_saved_arrays(dev, learner, saved)
    end
    cache.learner = learner
    invalidate_cache!(cache)
    return cache
end

# A learner with the structure and types of `fresh` (built for a compiled backend on
# `dev`) and the arrays of `saved` (a host copy): parameters, target parameters and
# optimizer moments. Numbers in optimizer states that the backend stores as device
# numbers (such as Adam's bias-correction powers) keep their fresh values.
function _copy_saved_arrays(dev, fresh, saved)
    is_node = x -> x isa AbstractArray || x isa Optimisers.Leaf
    return fmap(fresh, saved; exclude = x -> is_node(x) || Functors.isleaf(x)) do f, s
        if f isa Optimisers.Leaf
            return Optimisers.Leaf(f.rule, _copy_saved_arrays(dev, f.state, s.state), f.frozen)
        elseif f isa AbstractArray && s isa AbstractArray
            return dev(s)
        elseif f isa Number && s isa typeof(f)
            return s
        end
        return f
    end
end
