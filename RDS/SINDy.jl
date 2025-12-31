using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :NoiseRobustDifferentiation] .|> string
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
mae(x, y; dims = 2) = mean(abs, x - y; dims)
rmse(x, y; dims = 2) = sqrt.(mean(abs2, x - y; dims))
encode(x) = *(Char.(x)...)
decode(y) = Int64.(codepoint.(collect(y)))
DataFrame([rand(10) for _ in 1:5], :auto)
function add_diff(D::AbstractDataFrame; method = :FDM)
    dnames = "d" .* names(D)
    if method == :FDM
        return [DataFrame(diff(Matrix(D), dims = 1), dnames) D[1:(end-1), :]]
    else method == :TVD
        return [DataFrame([tvdiff(z, 10, 100, dx = 1) for z in eachcol(D)], dnames) D]
    end
end

@load "data/data_tnsr.jld2"
T = CSV.read("data/data_GLBy0.08_expt_93.0.csv", DataFrame)[:, 1]
I, X = eachcol(CSV.read("data/lon.csv", DataFrame))
J, Y = eachcol(CSV.read("data/lat.csv", DataFrame))

data = tensor2dataframe(tnsr)
dim = ncol(data)
zk = ["z$k" for k in 1:dim]
rename!(data, zk)
# ddata = add_diff(data)
@time ddata = add_diff(data, method = :TVD)

vrbl = (names(ddata)[1:dim], names(ddata)[(dim+1):end])
cnfg = cook(zk; poly = 0:2)

input = 30
output = 30
h = 1e-2
_T = findall(Year.(T) .< Year(2024))

results = DataFrame(Model = [], Hyperparameter = [], t0 = [], input = [], rmse1 = [], rmse7 = [], rmse30 = [])
for _ in 1:100
    selected = sort(cnfg[shuffle(1:nrow(cnfg))[1:10], :], :index)
    # encode(selected.index)
    # decode(encode(selected.index))
    t0 = rand(_T)
    t1 = t0 + input - 1
    f = SINDy(ddata[t0:t1, :], vrbl, selected)
    println("$(f.r2)")
    v = collect(ddata[t1, last(vrbl)])

    prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
    test = Matrix(data[t1:(t1+output), :])

    pfmc = rmse(prdt, test; dims = 2)[2:end]
    push!(results, ["SINDy", encode(selected.index), t0, input, pfmc[1], pfmc[7], pfmc[30]])
end

# results[:, [:rmse1, :rmse7, :rmse30]] .= round.(results[:, [:rmse1, :rmse7, :rmse30]], digits = 3)
# CSV.write("results.csv", results, bom = true)
# scatter(results.t0 .% 365, results.rmse1, yscale = :log10, ylims = [1e-1, 1e2])

tissue = sort(results, :rmse30)[1, :]
tissue_t1 = (tissue.t0 + tissue.input + 1)
g = SINDy(ddata[tissue.t0:tissue_t1, :], vrbl, cnfg[decode(tissue.Hyperparameter), :])
fttd = solve(g, collect(data[tissue_t1, :]), 0:h:output)[1:Int64(1/h):end, :]
test = Matrix(data[tissue_t1 .+ (0:output), last(vrbl)])
_fttd = reshape(fttd', 10, 15, :)
_test = reshape(test', 10, 15, :)
_rsdl = _fttd - _test
plot(sqrt.(mean.(abs2, eachslice(_rsdl, dims = 3))))
# surface(g.matrix)


# histogram([cnfg[[decode.(sort(results, :rmse1).Hyperparameter[95:end])...;], :vecv]...;], bin = 20)
# histogram([decode.(sort(results, :rmse1).Hyperparameter[1:5])...;], bin = 100)

plot(
    plot(ddata.z1),
    plot(diff(data.z1), yticks = [0]),
    scatter(ddata.z1, diff(data.z1)),
    layout = (3, 1), size = [800, 800], legend = :none
)


plot(
    plot(ddata.z1),
    plot(ddata.dz1, yticks = [0]),
    scatter(ddata.z1, ddata.dz1),
    layout = (3, 1), size = [800, 800], legend = :none
)

# t0_ = rand(_T, 10)
# results = DataFrame(Model = [], Hyperparameter = [], t0 = [], input = [], rmse1 = [], rmse7 = [], rmse30 = [])
# strain = []
# for _ in 1:10
#     vldn = zeros(nrow(cnfg))
#     @showprogress @threads for i in 1:nrow(cnfg)
#         selected = cnfg[[strain; i], :]
#         pfmc_ = zeros(1:10)
#         for j in 1:10
#             t_ = [(t0:(t0 + input - 1) for t0 in t0_[Not(j)])...;]
#             f = SINDy(ddata[t_, :], vrbl, selected)
#             t0 = t0_[j]
#             t1 = t0 + input - 1
#             # v = collect(ddata[t1, last(vrbl)])

#             # prdt = solve(f, v, 0:h:1)[1:Int64(1/h):end, :]
#             # test = Matrix(data[t1:(t1+1), :])
#             # pfmc_[j] = rmse(prdt, test; dims = 2)[2]
#             pfmc_[j] = f.mse
#         end
#         vldn[i] = mean(pfmc_)
#     end
#     push!(strain, argmin(vldn))
#     push!(results, ["SINDy", encode(strain), NaN, input, mean(vldn), NaN, NaN])
# end


# @time f = SINDy(shuffle(ddata)[1:150, :], vrbl, shuffle(cnfg)[1:50, :])
# f.recipe
# surface(f.matrix)

# heatmap(Matrix(ddata[:, 151:end]))
# heatmap(Matrix(ddata[:, 1:150]))
result = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [], Hyperparameter = [])
task0 = @showprogress @threads for t0 = _T
    t1 = t0 + input - 1
    r2_ = zeros(1000)
    hp_ = ["" for _ in 1:1000]
    for ti in eachindex(r2_)
        _selected = sort(cnfg[shuffle(1:nrow(cnfg))[1:10], :], :index)
        f = SINDy(ddata[t0:t1, :], vrbl, _selected)
        # println("$(f.r2)")
        r2_[ti] = f.r2
        hp_[ti] = encode(_selected.index)
    end
    
    selected = cnfg[decode(hp_[argmax(r2_)]), :]
    f = SINDy(ddata[t0:t1, :], vrbl, selected)

    v = collect(ddata[t1, last(vrbl)])
    prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
    test = Matrix(data[t1:(t1+output), :])

    pfmc = rmse(prdt, test; dims = 2)[2:end]
    push!(result, ["SINDy", t0, input, maximum(r2_), pfmc[1], pfmc[7], pfmc[30], encode(selected.index)])
end
result

istaskstarted(task0)
istaskdone(task0)
scatter(1 .- result.R2, result.rmse1, scale = :log10, xlabel = "1 - R2", ylabel = "RMSE at t = 1")
scatter(result.t0, result.rmse1, xlabel = "t0", ylabel = "RMSE1")
CSV.write("results.csv", result, bom = true)