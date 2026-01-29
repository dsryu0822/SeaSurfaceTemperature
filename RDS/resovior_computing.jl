using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :NoiseRobustDifferentiation,
    :Graphs, :Random] .|> string
try
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded")
catch e
    required = setdiff(packages, keys(Pkg.installed()))
    if !isempty(required) Pkg.add(required) end
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded after installation")
end

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

μ = mean(tnsr); σ = std(tnsr)
data = tensor2dataframe(tnsr)
# data = tensor2dataframe(tnsr)
dim = ncol(data)
zk = ["z$k" for k in 1:dim]
rename!(data, zk)

# id_observer = [1:10; 141:150; 20:10:140; 11:10:131]; # reshape(1:150, 10, 15)
id_observer = [1, 10, 141, 150]
# id_observer = [1]

M = length(id_observer)
P = 150 - M
N = 1000
D = 10
RN = erdos_renyi(N, D/N)
A = adjacency_matrix(RN) .* 2(rand(N, N) .- .5)
Win = (stack([shuffle([1; zeros(M-1)]) for _ in 1:N])') .* 2(rand(N, M) .- .5)
encode(r, u) = tanh.(A * r + Win * u)

U = Matrix(data[1:1000, id_observer])'
r_ = [randn(N)]
for u in eachcol(U)
    push!(r_, encode(r_[end], u))
end
pop!(r_); R = stack(r_)
S = Matrix(data[1:1000, Not(id_observer)])'
# Wout = S / R
Wout = S*R'inv(R*R' + 1e-3LinearAlgebra.I)
c = -vec(Wout * mean(R, dims = 2) - mean(S, dims = 2))
mean(abs2, Wout * R .+ c .- S)

decode(r, u) = Wout * encode(r, u) .+ c

[decode(r_[end], [data[1001, id_observer]...]) [data[1001, Not(id_observer)]...]]
scatter(decode(r_[end], [data[1001, id_observer]...]) - [data[1001, Not(id_observer)]...])

sqrt(mean(abs2, decode(r_[end], [data[1001, id_observer]...]) - [data[1001, Not(id_observer)]...]))

