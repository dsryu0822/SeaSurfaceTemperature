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
T = Date.(CSV.read("data/data_GLBy0.08_expt_93.0.csv", DataFrame)[:, 1])
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
h = 1e-2

t0 = rand(findall(T .> Date(2024))[1:(end-output)])
v = collect(ddata[t0, last(vrbl)])
history = vec(sum(abs2, Matrix(data) .- v', dims = 2))
history[Year.(T) .== Year(2024)] .= Inf
t0_ = []
for _ in 1:5
    push!(t0_, argmin(history))
    history[Year.(T) .== Year(T[t0_[end]])] .= Inf
end
plot(
    heatmap(X, Y, tnsr[:,:,t0]',     title = "$(T[t0])"),
    heatmap(X, Y, tnsr[:,:,t0_[1]]', title = "$(T[t0_[1]])"),
    heatmap(X, Y, tnsr[:,:,t0_[2]]', title = "$(T[t0_[2]])"),
    heatmap(X, Y, tnsr[:,:,t0_[3]]', title = "$(T[t0_[3]])"),
    heatmap(X, Y, tnsr[:,:,t0_[4]]', title = "$(T[t0_[4]])"),
    heatmap(X, Y, tnsr[:,:,t0_[5]]', title = "$(T[t0_[5]])"),
    size = [900, 600], layout = (2, 3), clims = extrema(tnsr[:, :, [t0; t0_]]),
)

folded = [ddata[t0 .+ (0:30), :] for t0 in t0_]
vldt = [Matrix(fold[:, last(vrbl)]) for fold in folded]
trng = [[folded[setdiff(1:5, k)]...;] for k in 1:5]
results = DataFrame(rmse1 = zeros(100), rmse7 = zeros(100), rmse30 = zeros(100), Hyperparameter = [encode(shuffle(1:nrow(cnfg))[1:10]) for _ in 1:100])
for i in 1:100
    println(i)
    selected = cnfg[decode(results.Hyperparameter[i]), :]
    pfmc1, pfmc7, pfmc30 = 0.0, 0.0, 0.0
    for j in 1:5
        prdt = mean(stack(vldt[Not(j)]), dims = 3)[:,:,1]
        # f = SINDy(trng[j], vrbl, selected)
        # v = vec(vldt[j][1,:])

        # prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
        pfmc = rmse(prdt, vldt[j]; dims = 2)[2:end]
        pfmc1 += pfmc[1]; pfmc7 += pfmc[7]; pfmc30 += pfmc[30]
    end
    results[i, 1:3] = ([pfmc1, pfmc7, pfmc30] ./ 5)
end
sort(results, :rmse1)

results

