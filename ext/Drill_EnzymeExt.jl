module Drill_EnzymeExt

using Enzyme: Enzyme, Active, Const, Duplicated, EnzymeRules
using Lux: AutoEnzyme
import Drill: value_and_gradient

# Enzyme differentiates a scalar function; the auxiliary output leaves through a Ref,
# written by a setter Enzyme treats as inactive.
_set_aux!(aux_ref, aux) = (aux_ref[] = aux; nothing)
EnzymeRules.inactive(::typeof(_set_aux!), ::Any...) = nothing

function _loss_only(f::F, x, aux_ref, args...) where {F}
    loss, aux = f(x, args...)
    _set_aux!(aux_ref, aux)
    return loss
end

# A Ref with the concrete type of `f(x, args...)[2]`
function _aux_ref(f::F, x, args...) where {F}
    R = Core.Compiler.return_type(f, Tuple{typeof(x), map(typeof, args)...})
    if R <: Tuple{Any, Any} && isconcretetype(fieldtype(R, 2))
        return Ref{fieldtype(R, 2)}()
    end
    _, aux = f(x, args...)
    return Ref{typeof(aux)}()
end

function _primal_mode(ad::AutoEnzyme)
    mode = Enzyme.ReverseWithPrimal
    if ad.mode !== nothing && Enzyme.runtime_activity(ad.mode)
        mode = Enzyme.set_runtime_activity(mode)
    end
    return mode
end

function value_and_gradient(ad::AutoEnzyme, f::F, x, args...) where {F}
    dx = Enzyme.make_zero(x)
    aux_ref = _aux_ref(f, x, args...)
    _, loss = Enzyme.autodiff(
        _primal_mode(ad), Const(_loss_only), Active,
        Const(f), Duplicated(x, dx), Const(aux_ref), map(Const, args)...,
    )
    return loss, aux_ref[], dx
end

end
