module LocalMethods

using SpecialFunctions: besselk, gamma

#M_vρσ is the matern covariance and G_σρ is Gaussian autocovariance model
export ℳ_ν, M_νρσ, G_σρ

tν𝒦t(t,ν) = t^ν * besselk(ν, t)

"""
    ℳ_ν(t, ν)

Compute (√(2ν)*t) ^ ν * 𝒦ν(√(2ν)*t) where 𝒦ν is the modified Bessel function of the second kind of order ν.

# Example
```julia-repl
julia> ℳ_ν(0.0, 0.5)
1.0
```
"""
function ℳ_ν(t, ν)
	pt, pν, p0, p1 = promote(t, ν, 0, 1)
	return (pt==p0) ? p1 : tν𝒦t(√(2pν)*pt,pν) * 2^(1-pν) / gamma(pν)
end

function M_νρσ(t; ν, ρ, σ)
	return σ * σ * ℳ_ν(t/ρ, ν)
end

function G_σρ(t; σ,ρ)
	return σ * σ *exp(-t^2/ρ^2)

end
end # end module
