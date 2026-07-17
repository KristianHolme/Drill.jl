function save_model_params_and_state(cache::RLCache, path::AbstractString; suffix::String = ".jld2")
    file_path = endswith(path, suffix) ? path : path * suffix
    host = cpu_device()
    save(
        file_path,
        Dict(
            "model" => cache.model,
            "parameters" => host(parameters(cache)),
            "states" => host(states(cache)),
            "train_state" => host(cache.train_state),
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
    if haskey(data, "train_state")
        cache.train_state = adapt(dev, data["train_state"])
    else
        ps = dev(data["parameters"])
        st = dev(data["states"])
        cache.train_state = _init_train_state(cache.model, alg, ps, st, identity)
    end
    invalidate_cache!(cache)
    return cache
end
