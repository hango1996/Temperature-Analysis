using PyPlot
using LinearAlgebra
using Statistics
using Random
using StaticArrays
import JLD2
using UKrig2


pwd()

##
# load data
elv_obs    = JLD2.FileIO.load("data/UStmax.jld2", "elv")
UStmax_obs = JLD2.FileIO.load("data/UStmax.jld2", "UStmax")
lon_obs    = JLD2.FileIO.load("data/UStmax.jld2", "lon")
lat_obs    = JLD2.FileIO.load("data/UStmax.jld2", "lat")

##
# load a grid of lat and lon with corresponding elevation
elv_grid      = JLD2.FileIO.load("data/krig_at.jld2", "elv_grid")
lat_grid_side = JLD2.FileIO.load("data/krig_at.jld2", "lat_grid_side")
lon_grid_side = JLD2.FileIO.load("data/krig_at.jld2", "lon_grid_side")

##
# plot
pygui(true)
figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=elv_obs, s=1)
title("elevation")
colorbar()

figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=UStmax_obs, s=1)
title("temperature")
colorbar()

figure(figsize=(10,5))
imshow(elv_grid,
    extent=(extrema(lon_grid_side)..., extrema(lat_grid_side)...),
)


## =========================================
# construct the krigin predictor
musr=3
ν=0.5
σg=1.0
σe=0.01
krig = generate_Gnu_krig(
	UStmax_obs,
	lon_obs,
	lat_obs;
	musr=musr,
	ν=ν,
	σg=σg,
	σe=σe,
)
est_UStmax_obs = krig.(lon_obs, lat_obs)

##
# plot the krig prediction at the observation points
figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=est_UStmax_obs, s=1)
title("Krig interpolation at the ")
colorbar()

figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=UStmax_obs, s=1)
title("observed temp ")
colorbar()


figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=est_UStmax_obs .- UStmax_obs, s=1)
title("resid")
colorbar()


## =========================================
#  plot the krig prediction on a grid of lon/lat points

# nlon = 600
# nlat = 200
# lon_extrema = extrema(lon_obs)
# lat_extrema = extrema(lat_obs)
# lon_grid_side = range(lon_extrema..., length=nlon)'
# lat_grid_side = reverse(range(lat_extrema..., length=nlat))

@time interp1_UStmax_grid = krig.(lon_grid_side, lat_grid_side)

figure(figsize=(10,5))
imshow(
	interp1_UStmax_grid,
	extent=(extrema(lon_grid_side)..., extrema(lat_grid_side)...),
    vmin=0, vmax=50
)
colorbar()


## ==============================
# Include elevation as a covariate

fdata = UStmax_obs
xdata = (lon_obs, lat_obs)
d = 2
m   = max(musr,floor(Int, ν))
n   = length(fdata)
monos, Fp = UKrig._construct_monos_Fp(m, Val(d))
mp = length(monos)

dmat = UKrig.distmat(xdata, xdata)
G₁₁  = (σg^2) .* Gnu.(dmat, ν)

FpVec = Fp.(xdata...)
F₁₁_monos   = reduce(hcat,permutedims(FpVec))

# now add a row that corresponds to elevation
#----------
n_elv_terms = 1
F₁₁ = vcat(F₁₁_monos, elv_obs')
#----------
# n_elv_terms = 2
# F₁₁ = vcat(F₁₁_monos, elv_obs', abs2.(elv_obs)')
#----------
mp_plus_elv = mp + n_elv_terms

Ξ   = [
	G₁₁ .+ σe^2*I(n)  F₁₁'
	F₁₁  zeros(mp_plus_elv, mp_plus_elv)
]
cb = Ξ \ vcat(fdata, zeros(mp_plus_elv))
c  = cb[1:length(fdata)]
b  = cb[length(fdata)+1:end]

fpb = UKrig._generate_fpb(monos, b[1:(end-n_elv_terms)], Val(d))
b_for_elev = b[(end-n_elv_terms+1):end]

let d=d, n=n, xdata=xdata, ν=ν, σg=σg, fpb=fpb, c=c
	global function krig_minus_elev(x::SVector{d,Q}) where Q<:Real
		sqdist = fill(Q(0),n)
		for i = 1:d
			sqdist .+= (x[i] .- xdata[i]).^2
		end
		Kvec = (σg^2) .* Gnu.(sqrt.(sqdist), ν)
		return dot(Kvec,c) + fpb(x)
	end
	global krig_minus_elev(x::Real...) = krig_minus_elev(SVector(x))
	global krig_minus_elev(x::NTuple{d,Q}) where {Q<:Real} = krig_minus_elev(SVector(x))
end

@time interp2_UStmax_grid_wo_elev = krig_minus_elev.(lon_grid_side, lat_grid_side)
interp2_UStmax_grid = copy(interp2_UStmax_grid_wo_elev)
interp2_UStmax_grid .+= b_for_elev[1] * elv_grid
#----------
interp2_UStmax_grid .+= b_for_elev[2] * abs2.(elv_grid)
#----------


figure(figsize=(10,5))
imshow(interp2_UStmax_grid,
    extent=(extrema(lon_grid_side)..., extrema(lat_grid_side)...),
    vmin=0, vmax=50
)
colorbar()



figure(figsize=(10,5))
imshow(interp2_UStmax_grid_wo_elev,
    extent=(extrema(lon_grid_side)..., extrema(lat_grid_side)...),
    vmin=0, vmax=50
)
colorbar()
