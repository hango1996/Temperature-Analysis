using CSV, DataFrames, PyPlot, LinearAlgebra
using RecipesBase

data_file = "/Users/hango/Desktop/UCDavis(2019-)/Fall 2019/STA250/Assignment/1/Dataset-1-UStmax/data/UStmax.csv"

# scans the file, determines the appropriate column types, stores it in a custom type
csv_data  = CSV.File(data_file)
df = DataFrame!(csv_data)  # puts it into a data frame

PyPlot()
figure(figsize=(10,10))
scatter(df.lon, df.lat, c=df.elev, s=1)
title("elevation")
colorbar()
PyPlot.show()
PyPlot.saves()
display()


figure(figsize=(10,5))
scatter(df.lon, df.lat, c=df.UStmax, s=1)
title("temperature")
colorbar()


K(x₁, x₂, y₁, y₂) = exp(-√((x₁-y₁)^2 + (x₂-y₂)^2))
lonlat = zip(df.lon, df.lat)
Σ = [K(x[1], x[2], y[1], y[2]) for x in lonlat, y in lonlat]

figure(figsize=(10,5))
scatter(df.lon, df.lat, c=Σ[:,1], s=1)
title("temperature")
colorbar()

dv = eigen(Σ)
dv.values
dv.vectors

ch = cholesky(Σ)
ch.L
ch.U




using PyPlot
x = 1:10; y = rand(10); # These are the plotting data
#plotly()
plot(x,y)
