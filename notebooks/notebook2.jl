#########################################
#
## module
#
#########################################
module LocalMethods

using SpecialFunctions: besselk, gamma

export ℳ_ν, M_νρσ

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

end # end module

#########################################
#
## global scope
#
#########################################


using .LocalMethods
using CSV
using DataFrames
using PyPlot
using LinearAlgebra
using Statistics
using Random

## load data
data_file = "/Users/hango/Desktop/UCDavis(2019-)/Fall 2019/STA250/Assignment/1/Dataset-1-UStmax/data/UStmax.csv"
csv_data  = CSV.File(data_file)
df        = DataFrame!(csv_data)  # puts it into a data DataFrame

## Check that Matern methods are inferred correctly
@code_warntype ℳ_ν(0, 0.25)
@code_warntype ℳ_ν(0.0, 0.25)
@code_warntype ℳ_ν(0.1, 1//2)
@code_warntype M_νρσ(0.0; ν=1, ρ=0.9, σ=1)
@code_warntype M_νρσ(0.1; ν=1.0, ρ=0.9, σ=1)


## plot Matern
pygui(true)
figure()
x = range(0, 3, length=300)
plot(x, ℳ_ν.(x, 0.25), label=L"\nu=0.25")
plot(x, ℳ_ν.(x, 0.5), label=L"\nu=0.5")
plot(x, ℳ_ν.(x, 2.0), label=L"\nu=1.0")
plot(x, ℳ_ν.(x, 2.5), label=L"\nu=2.0")
xlabel("lag t")
title("Matern auto-covariance")
legend()

figure()
x = range(0, 3, length=300)
plot(x, 2 .* (ℳ_ν.(0, 0.25) .- ℳ_ν.(x, 0.25)), label=L"\nu=0.25")
plot(x, 2 .* (ℳ_ν.(0, 0.5) .- ℳ_ν.(x, 0.5)), label=L"\nu=0.5")
plot(x, 2 .* (ℳ_ν.(0, 2.0) .- ℳ_ν.(x, 2.0)), label=L"\nu=1.0")
plot(x, 2 .* (ℳ_ν.(0, 2.5) .- ℳ_ν.(x, 2.5)), label=L"\nu=2.0")
xlabel("lag t")
title("Matern variogram")
legend()

#variogram means


## construct cov matrix
nrm = sqrt.((df.lon .- df.lon').^2 .+ (df.lat .- df.lat').^2)

Σ = M_νρσ.(nrm; ν=1.5, ρ=1.0, σ=1.0) |> Symmetric
Σλs = eigen(Σ).values
figure(9)
sort(Σλs)[1:end÷10] |> plot

figure()
Σ = M_νρσ.(nrm; ν=2.5, ρ=10.0, σ=1.0) |> Symmetric
Σλs = eigen(Σ).values
sort(Σλs)[1:end÷10] |> plot


Σ = M_νρσ.(nrm; ν=4.5, ρ=10.0, σ=1.0) |> Symmetric
Σλs = eigen(Σ).values
figure(9)
sort(Σλs)[1:end÷10] |> plot



## Lets try to whiten the data and Σ to reduce the dynamic range
proj_out_basis = hcat(1 .+ 0 .* df.lon, df.lon, df.lat, df.lon.^2, df.lat.^2, df.lon .* df.lat)
Δ = nullspace(proj_out_basis')

Σ = M_νρσ.(nrm; ν=4.5, ρ=10.0, σ=1.0) |> Symmetric
Σ′ = Symmetric(transpose(Δ) * (Σ * Δ))
Σ′λs = eigen(Σ′).values
figure(10)
sort(Σ′λs)[1:end÷10] |> plot

##
non_diag_distances_sorted = sort(vec(nrm))[size(df,1)+1:end]
findall(nrm .== non_diag_distances_sorted[1])
findall(nrm .== non_diag_distances_sorted[3])
ct_ind = setdiff(Set(1:size(df,1)), Set([656, 739, 1361])) |> collect |> sort
nrm′ = nrm[ct_ind, ct_ind]

Σ′ = M_νρσ.(nrm′; ν=4.5, ρ=10.0, σ=1.0) |> Symmetric
Σ′λs = eigen(Σ′).values
figure()
sort(Σ′λs)[1:end÷10] |> plot



# final couple notes
using BenchmarkTools
Σ = M_νρσ.(nrm; ν=2.5, ρ=10.0, σ=1.0) |> Symmetric
d = randn(size(Σ,2))
invΣd1 = inv(Σ) * d
invΣd2 = Σ \ d
@benchmark inv(Σ) * d
@benchmark Σ \ d

std(d - Σ * (Σ \ d))
std(d - Σ * (inv(Σ) * d))

ch = cholesky(Σ)
@benchmark inv(ch.L) * d
@benchmark ch.L \ d

dot(d, Σ \ d)
dot(ch.L \ d, ch.L \ d)
dot(d, inv(Σ) * d)

ch.L*ch.L'

#Project
