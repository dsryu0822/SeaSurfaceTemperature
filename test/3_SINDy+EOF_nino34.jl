include("0_utils.jl")
include("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core/header.jl")


@load "G:/seasurface/seasurface_nino3.4/data_tnsr.jld2"
T = CSV.read("data/data_GLBy0.08_expt_93.0.csv", DataFrame)[:, 1]
I, X = eachcol(CSV.read("data/lon.csv", DataFrame))
J, Y = eachcol(CSV.read("data/lat.csv", DataFrame))

data = tensor2dataframe(tnsr)
Q = ncol(data)
rename!(data, ["z$k" for k in 1:Q])

names(eachcol(data))

eltype.(eachcol(data))
Float64.(eachcol(data)[1])
missing ∈ eachcol(data)[6]

eof = EOF(Matrix(data)); # rmse(Matrix(data), eof(eof.U))



eltype.(eachcol(data))

bit_sea = .!ismissing.(tnsr[:, :, 1])

heatmap(bit_sea', size = [600, 300])