@load "data/data_tnsr.jld2"
T = CSV.read("data/data_GLBy0.08_expt_93.0.csv", DataFrame)[:, 1]
I, X = eachcol(CSV.read("data/lon.csv", DataFrame))
J, Y = eachcol(CSV.read("data/lat.csv", DataFrame))

μ = mean(tnsr); σ = std(tnsr)
data = tensor2dataframe(tnsr)
# data = tensor2dataframe(tnsr)
Q = ncol(data)
zk = ["z$k" for k in 1:Q]
rename!(data, zk)