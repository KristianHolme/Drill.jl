# Helper functions for building policies

abstract type AbstractWeightInitializer end

struct OrthogonalInitializer{T <: AbstractFloat} <: AbstractWeightInitializer
    gain::T
end

function (init::OrthogonalInitializer{T})(rng::AbstractRNG, out_dims::Int, in_dims::Int) where {T}
    return orthogonal(rng, T, out_dims, in_dims; gain = init.gain)
end

function get_feature_extractor(O::Box)
    if ndims(O) == 1
        return Lux.NoOpLayer()
    end
    return Lux.FlattenLayer()
end

function get_feature_extractor(O::Discrete)
    if ndims(O) == 1
        return Lux.NoOpLayer()
    end
    return Lux.FlattenLayer()
end

function get_mlp(
        latent_dim::Int, output_dim::Int, hidden_dims::Vector{Int}, activation::Function,
        bias_init, hidden_init, output_init
    )
    layers = []
    if isempty(hidden_dims)
        push!(layers, Dense(latent_dim, 1, init_weight = output_init, init_bias = bias_init))
    else
        push!(
            layers, Dense(
                latent_dim, hidden_dims[1], activation, init_weight = hidden_init,
                init_bias = bias_init
            )
        )
        for i in 2:length(hidden_dims)
            push!(
                layers, Dense(
                    hidden_dims[i - 1], hidden_dims[i], activation,
                    init_weight = hidden_init, init_bias = bias_init
                )
            )
        end
        push!(
            layers, Dense(
                hidden_dims[end], output_dim, init_weight = output_init,
                init_bias = bias_init
            )
        )
    end
    return Chain(layers...)
end

function get_actor_head(
        latent_dim::Int, action_dim::Int, hidden_dims::Vector{Int},
        activation::Function, bias_init, hidden_init, output_init
    )
    return get_mlp(
        latent_dim, action_dim, hidden_dims, activation, bias_init, hidden_init,
        output_init
    )
end

function get_actor_head(
        latent_dim::Int, A::Box, hidden_dims::Vector{Int},
        activation::Function, bias_init, hidden_init, output_init
    )
    chain = get_actor_head(
        latent_dim, prod(size(A)), hidden_dims, activation, bias_init,
        hidden_init, output_init
    )
    chain = Chain(chain, ReshapeLayer(size(A)))
    return chain
end

function get_actor_head(
        latent_dim::Int, A::Discrete, hidden_dims::Vector{Int},
        activation::Function, bias_init, hidden_init, output_init
    )
    chain = get_actor_head(
        latent_dim, A.n, hidden_dims, activation, bias_init,
        hidden_init, output_init
    )
    return chain
end

function get_q_mlp_tuple_input(
        latent_dim::Int, action_dim::Int, hidden_dims::Vector{Int},
        activation::Function, bias_init, hidden_init, output_init
    )
    # First layer: Parallel(connection, Dense_f, Dense_a) on (feats, actions).
    # connection(a, b) = activation.(a .+ b). Dense_f has bias, Dense_a has zero bias (vcat equivalence).
    connection(a, b) = activation.(a .+ b)
    zeros_bias = (rng, bias_len) -> zeros(Float32, bias_len)
    if isempty(hidden_dims)
        first_layer = Lux.Parallel(
            connection,
            Lux.Dense(
                latent_dim, 1; init_weight = output_init,
                init_bias = bias_init
            ),
            Lux.Dense(
                action_dim, 1; init_weight = output_init,
                init_bias = zeros_bias
            ),
        )
        return Lux.Chain(first_layer)
    end
    out_dim = hidden_dims[1]
    first_layer = Lux.Parallel(
        connection,
        Lux.Dense(
            latent_dim, out_dim, identity;
            init_weight = hidden_init, init_bias = bias_init
        ),
        Lux.Dense(
            action_dim, out_dim, identity;
            init_weight = hidden_init, init_bias = zeros_bias
        ),
    )
    rest_layers = []
    for i in 2:length(hidden_dims)
        push!(
            rest_layers,
            Lux.Dense(
                hidden_dims[i - 1], hidden_dims[i], activation;
                init_weight = hidden_init, init_bias = bias_init
            ),
        )
    end
    push!(
        rest_layers,
        Lux.Dense(
            hidden_dims[end], 1; init_weight = output_init,
            init_bias = bias_init
        ),
    )
    return Lux.Chain(first_layer, Lux.Chain(rest_layers...))
end

function get_critic_head(
        latent_dim::Int, action_space::Box, hidden_dims::Vector{Int},
        activation::Function, bias_init, hidden_init, output_init, critic_type::QCritic
    )
    action_dim = size(action_space) |> prod
    chains = [
        get_q_mlp_tuple_input(
            latent_dim, action_dim, hidden_dims, activation, bias_init,
            hidden_init, output_init
        ) for _ in 1:critic_type.n_critics
    ]
    if length(chains) == 1
        return chains[1]
    end
    return Lux.BranchLayer(chains...; fusion = vcat)
end

function get_critic_head(
        latent_dim::Int, action_space::AbstractSpace,
        hidden_dims::Vector{Int}, activation::Function, bias_init, hidden_init, output_init,
        critic_type::VCritic
    )
    return get_mlp(latent_dim, 1, hidden_dims, activation, bias_init, hidden_init, output_init)
end

function zero_critic_grads!(critic_grad::NamedTuple, layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SharedFeatures})
    if haskey(critic_grad, :critic_head)
        fmap(critic_grad.critic_head) do x
            if x isa AbstractArray
                x .= zero(eltype(x))
            end
            return x
        end
    end
    return nothing
end

function zero_critic_grads!(critic_grad::NamedTuple, layer::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, SeparateFeatures})
    if haskey(critic_grad, :critic_head)
        fmap(critic_grad.critic_head) do x
            if x isa AbstractArray
                x .= zero(eltype(x))
            end
            return x
        end
    end
    if haskey(critic_grad, :critic_feature_extractor)
        fmap(critic_grad.critic_feature_extractor) do x
            if x isa AbstractArray
                x .= zero(eltype(x))
            end
            return x
        end
    end
    return nothing
end
