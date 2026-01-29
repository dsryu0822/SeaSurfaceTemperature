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
# encode(x) = *(Char.(x)...)
# decode(y) = Int64.(codepoint.(collect(y)))
DataFrame([rand(10) for _ in 1:5], :auto)
function add_diff(D::AbstractDataFrame; method = :FDM)
    dnames = "d" .* names(D)
    if method == :FDM
        return [DataFrame(diff(Matrix(D), dims = 1), dnames) D[1:(end-1), :]]
    else method == :TVD
        return [DataFrame([tvdiff(z, 10, 100, dx = 1) for z in eachcol(D)], dnames) D]
    end
end
function frequency(inventory)
    freq = Dict{eltype(inventory), Int64}()
    for item in inventory
        freq[item] = get(freq, item, 0) + 1
    end
    return freq
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

"""'''''''''''''''''''''''''''''''''''''''''''''

                brute-force search              

'''''''''''''''''''''''''''''''''''''''''''''"""

cnfg = cook(zk; poly = 0:2)
result = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [], Hyperparameter = [])
task0 = @async begin
    @showprogress @threads for t0 = _T
    t1 = t0 + input - 1
    r2_ = zeros(1000)
    hp_ = [[] for _ in 1:1000]
    for ti in eachindex(r2_)
        _selected = sort(cnfg[shuffle(1:nrow(cnfg))[1:5], :], :index)
        f = SINDy(ddata[t0:t1, :], vrbl, _selected)
        # println("$(f.r2)")
        r2_[ti] = f.r2
        hp_[ti] = _selected.index
    end
    
    selected = cnfg[hp_[argmax(r2_)], :]
    f = SINDy(ddata[t0:t1, :], vrbl, selected)

    v = collect(ddata[t1, last(vrbl)])
    prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
    test = Matrix(data[t1:(t1+output), :])

    pfmc = rmse(prdt, test; dims = 2)[2:end]
    push!(result, ["SINDy", t0, input, maximum(r2_), pfmc[1], pfmc[7], pfmc[30], selected.index])
    end
result
CSV.write("dashboard/SINDy_brute_force.csv", result, bom = true)
end