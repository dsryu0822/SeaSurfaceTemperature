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
add_next(D::AbstractDataFrame) = [DataFrame(Matrix(D[2:end, :]), "d" .* names(D)) D[1:(end-1), :]]
mae(x, y; dims = 2) = mean(abs, x - y; dims)
rmse(x, y; dims = 2) = sqrt.(mean(abs2, x - y; dims))
encode(x) = *(Char.(x)...)
decode(y) = Int64.(codepoint.(collect(y)))
function maptomap(f, v, tspan)
    V = [v]
    for t in tspan[2:end]
        push!(V, f(V[end], t))
    end
    return stack(V)
end
maptomap(g, v, 0:30)

@load "data/data_tnsr.jld2"
T = CSV.read("data/data_GLBy0.08_expt_93.0.csv", DataFrame)[:, 1]
I, X = eachcol(CSV.read("data/lon.csv", DataFrame))
J, Y = eachcol(CSV.read("data/lat.csv", DataFrame))

data = tensor2dataframe(tnsr)
dim = ncol(data)
zk = ["z$k" for k in 1:dim]
rename!(data, zk)
ddata = add_next(data)

vrbl = (names(ddata)[1:dim], names(ddata)[(dim+1):end])
cnfg = cook(zk; poly = 0:2)

input = 30
output = 30
h = 1e-2

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

results

results[:, [:rmse1, :rmse7, :rmse30]] .= round.(results[:, [:rmse1, :rmse7, :rmse30]], digits = 3)
CSV.write("results.csv", results, bom = true)
scatter(results.t0 .% 365, results.rmse1, yscale = :log10, ylims = [1e-1, 1e2])

tissue = sort(results, :rmse30)[50, :]
tissue_t1 = (tissue.t0 + tissue.input + 1)
g = SINDy(ddata[tissue.t0:tissue_t1, :], vrbl, cnfg[decode(tissue.Hyperparameter), :])
fttd = solve(g, collect(data[tissue_t1, :]), 0:h:output)[1:Int64(1/h):end, :]
test = Matrix(data[tissue_t1 .+ (0:output), last(vrbl)])
_fttd = reshape(fttd', 10, 15, :)
_test = reshape(test', 10, 15, :)
_rsdl = _fttd - _test
plot(sqrt.(mean.(abs2, eachslice(_rsdl, dims = 3))))
surface(g.matrix)

t = 19
plot(
    heatmap(Y, X, _test[:, :, t]),
    heatmap(Y, X, _fttd[:, :, t]),
    heatmap(Y, X, _rsdl[:, :, t], color = :balance),
    size = [900, 300], layout = (1, 3),
)



sample1 = sort(results, :rmse1)[1, :]
sample7 = sort(results, :rmse7)[1, :]
sample3 = sort(results, :rmse30)[1, :]

[cnfg[decode(sample1.Hyperparameter), :].vecv...;]
[cnfg[decode(sample7.Hyperparameter), :].vecv...;]
[cnfg[decode(sample3.Hyperparameter), :].vecv...;]
results
g.r2
plot(ddata.z1)
plot(ddata.dz1)
plot(ddata.z10[1:30])
plot(ddata.z10[1:30])
plot(ddata.z10[31:60])
plot(ddata.z10[1:30])

selected = sort(cnfg[shuffle(1:nrow(cnfg))[1:10], :], :index)
g = SINDy(ddata, vrbl, selected)
