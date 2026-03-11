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

        Empirical orthogonal function (EOF)

'''''''''''''''''''''''''''''''''''''''''''''"""
M = Matrix(data)
U, S, V = svd(Matrix(data))
(S.^2) ./ sum(S .^ 2)
(S) ./ sum(S)
plot(S, seriestype=:scatter, title="Scree Plot", xlabel="Mode", ylabel="Singular Value", legend=false)
svd(Matrix(ddata[:, 1:dim]))
plot(U[:, 2])
# plot(data.z1)
V

eof = DataFrame(u1 = U[:, 1], u2 = U[:, 2], u3 = U[:, 3])
deof = add_diff(eof, method = :TVD)
uvrbl = (names(deof)[1:ncol(eof)], names(deof)[(ncol(eof)+1):end])
ucnfg = cook(last(uvrbl); poly = 0:3)
fu = SINDy(deof, uvrbl, ucnfg)
fu.r2


CSV.read("EOF_modes_anomaly.csv", DataFrame)
CSV.read("PCs_anomaly_EOF.csv", DataFrame)