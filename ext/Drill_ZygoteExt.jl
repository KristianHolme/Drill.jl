module Drill_ZygoteExt

using Lux: AutoZygote
using Zygote: Zygote
import Drill: value_and_gradient

function value_and_gradient(::AutoZygote, f::F, x, args...) where {F}
    (loss, aux), back = Zygote.pullback(p -> f(p, args...), x)
    grad = only(back((one(loss), nothing)))
    return loss, aux, grad
end

end
