using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2] .|> string
try
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded")
catch e
    required = setdiff(packages, keys(Pkg.installed()))
    if !isempty(required) Pkg.add(required) end
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded after installation")
end
include("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core/header.jl")

tensor2dataframe(tnsr) = DataFrame(stack(vec(eachslice(tnsr, dims = (1,2)))), :auto)
add_diff(D::AbstractDataFrame) = [DataFrame(diff(Matrix(D), dims = 1), "d" .* names(D)) D[1:(end-1), :]]
mae(x, y; dims = 2) = mean(abs, x - y; dims)
rmse(x, y; dims = 2) = sqrt.(mean(abs2, x - y; dims))
encode(x) = *(Char.(x)...)
decode(y) = Int64.(codepoint.(collect(y)))


@load "data/data_tnsr.jld2"
T = CSV.read("data/data_GLBy0.08_expt_93.0.csv", DataFrame)[:, 1]
I, X = eachcol(CSV.read("data/lon.csv", DataFrame))
J, Y = eachcol(CSV.read("data/lat.csv", DataFrame))

data = tensor2dataframe(tnsr)
dim = ncol(data)
zk = ["z$k" for k in 1:dim]
rename!(data, zk)
ddata = add_diff(data)

vrbl = (names(ddata)[1:dim], names(ddata)[(dim+1):end])
cnfg = cook(zk; poly = 0:2)

trial = 1000
input = 30
output = 30
h = 1e-2
_T = findall(Year.(T) .< Year(2024))

results = DataFrame(t0 = zeros(Int64, trial), rmse1 = zeros(trial), rmse7 = zeros(trial), rmse30 = zeros(trial), Hyperparameter = [encode(sort(shuffle(1:nrow(cnfg))[1:10])) for _ in 1:trial])
@showprogress @threads for i in 1:trial
    selected = sort(cnfg[shuffle(1:nrow(cnfg))[1:10], :], :index)
    # encode(selected.index)
    # decode(encode(selected.index))
    t0 = rand(_T)
    t1 = t0 + input - 1
    f = SINDy(ddata[t0:t1, :], vrbl, selected)
    # println("$(f.r2)")
    v = collect(ddata[t1, last(vrbl)])

    prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
    test = Matrix(data[t1:(t1+output), :])

    pfmc = rmse(prdt, test; dims = 2)[2:end]
    results[i, 1:4] .= [t0, pfmc[1], pfmc[7], pfmc[30]]
end
sort(results, :rmse1)
top20 = []
termnumber = [decode.(sort(results, :rmse1).Hyperparameter[1:10])...;]
unique(termnumber)
unique([cnfg[termnumber, :vecv]...;])
push!(top20, modes(termnumber))
termnumber = [decode.(sort(results, :rmse7).Hyperparameter[1:20])...;]
push!(top20, modes(termnumber))
termnumber = [decode.(sort(results, :rmse30).Hyperparameter[1:20])...;]
push!(top20, modes(termnumber))
SINDy(ddata[72 .+ (0:output), :], vrbl, cnfg[[top20...;], :])


scatter(results.t0 .% 365, results.rmse1, yscale = :log10, ylims = [1e-1, 1e2])
tissue = sort(results, :rmse30)[1,:]
selected = cnfg[decode(sort(results, :rmse1).Hyperparameter[1]), :]
t0 = tissue.t0
t1 = t0 + input - 1
g = SINDy(ddata[t0:t1, :], vrbl, selected)

fttd = solve(g, collect(data[t1, :]), 0:h:output)[1:Int64(1/h):end, :]
test = Matrix(data[t1 .+ (0:output), last(vrbl)])
_fttd = reshape(fttd', 10, 15, :)
_test = reshape(test', 10, 15, :)
_rsdl = _fttd - _test
plot(sqrt.(mean.(abs2, eachslice(_rsdl, dims = 3))))
clims = [_test; _fttd] |> extrema

t = 1
plot(
    heatmap(Y, X, _test[:, :, t]; clims),
    heatmap(Y, X, _fttd[:, :, t]; clims),
    heatmap(Y, X, _rsdl[:, :, t], color = :balance),
    size = [900, 300], layout = (1, 3)
)


results2 = DataFrame(t0 = zeros(Int64, 2trial), rmse1 = zeros(2trial), rmse7 = zeros(2trial), rmse30 = zeros(2trial), Hyperparameter = [encode(sort(shuffle(1:nrow(cnfg))[1:10])) for _ in 1:2trial])
selected = cnfg[decode(sort(results, :rmse1).Hyperparameter[1]), :]
@showprogress @threads for t0 in 1:(2trial)
    t1 = t0 + input - 1
    f = SINDy(ddata[t0:t1, :], vrbl, selected)
    v = collect(ddata[t1, last(vrbl)])

    prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
    test = Matrix(data[t1:(t1+output), :])

    pfmc = rmse(prdt, test; dims = 2)[2:end]
    results2[t0, 1:4] .= [t0, pfmc[1], pfmc[7], pfmc[30]]
end

scatter(results.rmse1, yscale = :log10, ylims = [1e-1, 1e2])