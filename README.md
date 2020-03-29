# Dataset 1

Mean summer daily maximum temperature for June-August, 1990 at 4408 observation locations in the United States (National Center for Atmospheric Research). Data file `data/UStmax.csv` has four columns corresponding to latitude, longitude, elevation and mean daily maximum temperature.

## Project 1


*Summary:* Model the mean daily maximum temperature as a Gaussian random field (GRF) with Matérn autocovariance. Fit the Matérn parameters `ν, ρ, σ²` using the (GRF) likelihood and use the fitted values for random field prediction (on a dense grid of spatial points over the US) and point-wise uncertainty quantification in terms of the conditional standard deviation of prediction. Compare with bilinear interpolation.

*Specific details:*

1) First subtract the global mean temperature from all the data points.

2) Then remove a random subset of the mean centered data observations (10% of the data should be fine) and reserve them for a test set to measure quality of prediction at the end. Also use them to test your prediction uncertainty quantification.

3) When fitting the Matérn parameters you can restrict `ν ∈ {1/2, 3/2, 5/2, 7/2, ...}`. Also keep in mind that there will be a flat ridge in the loglikelihood for each fixed `ν` which should corresponds to `σ² = constant * ρ^(2ν)` so finding a strictly optimal set of parameters `ν, ρ, σ²` will be quite difficult. I am looking for a reasonable choice of `ν, ρ, σ²` that has a loglikelihood value which is arguably `± 𝒪(10)` of the MLE.

4) Besides comparing Matérn GRF prediction and bilinear interpolation, also include a comparison when using the Gaussian autocovariance model `K(t) = σ² exp(-t²/ρ²)`. Summarize any numerical issues you encounter.

5) Write up a report which summaries your findings and submit in either pdf form or jupyter notebook form. This report should include code snippets, images showing the spatial prediction maps and summaries of any numerical issues you encounter.

6) You can see the report in the file `notebooks/Assignment1.ipynb`

## Project 2
*Summary:* Use Kriging to construct a high resolution prediction of mean daily maximum temperature on the (lat, lon) grid given in `data/krig_at.jld2`. Use the generalized covariance function which is the principle irregular term from our Matérn covariance model. For the Kriging covariates, use all monomials (in lat and lon) of order ≤ 2 and also include elevation as a covariate. Use the REML loglikelihood to find a good value for the covariance and noise parameters, `ν, σg, σε`, where `ν ∈ {1/2, 3/2}`, `σg > 0` and the nugget `σε > 0`. Use the estimated parameters to make your Kriging prediction.
*Notes:*
1) Here is some Julia code for the generalized covariance function that matches the principle irregular term from our Matérn covariance model.
```
function Gnu(t::A, ν::B) where {A<:Real, B<:Real}
C = promote_type(A,B)
if t==A(0)
return C(0)
end
if floor(ν)==ν
scν = (2 * (-1)^ν) * (- (ν/2)^ν / gamma(ν) / gamma(ν+1))
return C(scν * t^(2ν) * log(t))
else
scv = (π / sin(ν*π)) * (- (ν/2)^ν / gamma(ν) / gamma(ν+1))
return C(scν * t^(2ν))
end
end
```
2) To fit the Kriging model you have the elevation at each observation point in the file ``. To make the Kriging prediction you will need the elevation corresponding to each (lat,lon) grid point. The gridded elevations can be found in `data/krig_at.jld2`. Take a look at `notebooks/notebook3.jl` to find some code for loading and plotting.

3) We summaries the report in `notebook/Assignment2.ipynb`


# Repository Description 
### [/data](data)
[USTmax.csv](data/UStmax.csv): Mean summer daily maximum temperature for June-August, 1990 at 4408 obervation locations in the United States(National Center for Atmospheric Research). it has four columns corresponding to latitude, longitude, elevation and mean daily maximum temperature

### [code](code)
[LocalMethods.jl](code/LocalMethods.jl): Create local methods 

[UKrig2.jl](code/UKrig2.jl): create 

### [/notebooks](notebooks)
[Assignment1.ipynb](notebooks/Assignment1.ipynb):  Use matern covariance to analyze the data and report 

[Assignment2.ipynb](notebooks/Assignment2.ipynb):  Use KrigY model and generalized covariance function to analyze the data and report 
