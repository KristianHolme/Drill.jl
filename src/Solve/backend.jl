# How learner updates run. Updates are pure functions; the device of the learner's
# parameters decides how they are executed. On the CPU they are plain Julia calls;
# Drill_ReactantExt compiles them once per argument shape for a ReactantDevice.

"""
    run_update(dev, cache, update, args...) -> (learner, metrics)

Run the pure update function `update(args...)` for parameters on `dev`.
"""
run_update(::AbstractDevice, cache, update::F, args...) where {F} = update(args...)

"""
    gradient_backend(dev, ad)

The AD backend used inside updates for parameters on `dev`. Defaults to `ad`, the
`ad_type` passed to `init`.
"""
gradient_backend(::AbstractDevice, ad) = ad

"""
    requires_fixed_shapes(dev) -> Bool

Whether updates on `dev` need every minibatch to have the same size, as compiled
backends do. When `true`, the last partial minibatch of an epoch is dropped.
"""
requires_fixed_shapes(::AbstractDevice) = false

"""
    host_metrics(metrics::NamedTuple) -> NamedTuple

The metrics of an update as `Float32` host scalars.
"""
host_metrics(metrics::NamedTuple) = map(_host_scalar, metrics)

_host_scalar(x::Number) = Float32(x)
_host_scalar(x::AbstractArray) = Float32(only(cpu_device()(x)))

"""
    device_rng(dev, rng)

The RNG used inside updates for parameters on `dev`: `rng` itself on the CPU, a device
RNG seeded from it on a compiled backend.
"""
device_rng(::AbstractDevice, rng) = rng
