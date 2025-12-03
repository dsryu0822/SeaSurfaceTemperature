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

input = 30
output = 30
h = 1e-1

results = DataFrame(Model = [], Hyperparameter = [], t0 = [], input = [], rmse1 = [], rmse7 = [], rmse30 = [])
for _ in 1:100
    try
    selected = sort(cnfg[shuffle(1:nrow(cnfg))[1:10], :], :index)
    # encode(selected.index)
    # decode(encode(selected.index))
    t0 = rand(eachindex(T))
    t1 = t0 + input - 1
    f = SINDy(ddata[t0:t1, :], vrbl, selected)
    v = collect(ddata[t1, last(vrbl)])

    prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
    test = Matrix(data[t1:(t1+output), :])

    pfmc = rmse(prdt, test; dims = 2)[2:end]
    push!(results, ["SINDy", encode(selected.index), t0, input, pfmc[1], pfmc[7], pfmc[30]])
    catch e
        print(".")
    end
end
sort(results, :rmse30)

results[:, [:rmse1, :rmse7, :rmse30]] .= round.(results[:, [:rmse1, :rmse7, :rmse30]], digits = 3)
CSV.write("results.csv", results, bom = true)
scatter(results.t0 .% 365, results.rmse1, yscale = :log10, ylims = [1e-1, 1e2])


tissue = sort(results, :rmse1)[1, :]
g = SINDy(ddata[tissue.t0:(tissue.t0 + tissue.input + 1), :], vrbl, cnfg[decode(tissue.Hyperparameter), :])
fttd = solve(g, collect(data[(tissue.t0 + tissue.input + 1), :]), 0:h:output)[1:Int64(1/h):end, :]
test = Matrix(data[(tissue.t0 + tissue.input + 1) .+ (0:output), last(vrbl)])
_fttd = reshape(fttd', 10, 15, :)
_test = reshape(test', 10, 15, :)
_rsdl = _fttd - _test

plot(
    heatmap(Y, X, _test[:, :, 6]),
    heatmap(Y, X, _fttd[:, :, 6]),
    heatmap(Y, X, _rsdl[:, :, 6], color = :balance),
    size = [900, 300], layout = (1, 3),
)

heatmap(reshape(fttd', 10, 15, :)[:,:,1])
heatmap(reshape(collect(data[21, :])', 10, 15))
surface(reshape(collect(data[29, :])', 10, 15), alpha = .5)
plot(data.z1)

decode("ஏಹᆰᙩᴏἷ†⒩➰⯕")