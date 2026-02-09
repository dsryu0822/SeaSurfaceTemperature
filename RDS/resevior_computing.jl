using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :NoiseRobustDifferentiation,
    :Graphs, :Random, :LinearAlgebra] .|> string
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
Q = ncol(data)
zk = ["z$k" for k in 1:Q]
rename!(data, zk)

# id_observer = [1:10; 141:150; 20:10:140; 11:10:131]; # reshape(1:150, 10, 15)
id_observer = [1, 10, 141, 150]
# id_observer = [1]

M = length(id_observer)
P = Q - M
α = 1
ρ = 5
β = 1e-2
N = 500
D = 2
RN = erdos_renyi(N, (D*N ÷ 2))
A = adjacency_matrix(RN) .* 2(rand(N, N) .- .5)
A = ρ*A / maximum(sqrt.(sum.(abs2, eigen(Matrix(A)).values)))

Win = sparse(stack([shuffle([1; zeros(M-1)]) for _ in 1:N])') .* 2(rand(N, M) .- .5)
encode(r, u) = (1-α)*r + tanh.(A * r + Win * u)

U = Matrix(data[1:150, id_observer])'
r_ = [randn(N)]
for u in eachcol(U)
    push!(r_, encode(r_[end], u))
end
pop!(r_); R = stack(r_)
S = Matrix(data[1:150, Not(id_observer)])'
# Wout = S / R
Wout = S*R'inv(R*R' + (β*LinearAlgebra.I))
c = -vec(Wout * mean(R, dims = 2) - mean(S, dims = 2))
mean(abs2, Wout * R .+ c .- S)

decode(r, u) = Wout * encode(r, u) .+ c

actl = [data[151, Not(id_observer)]...]
prdt = decode(r_[end], [data[151, id_observer]...])
only(rmse(prdt', actl'))


struct prct1
    M::Int64
    P::Int64
    N::Int64
    α::Float64
    β::Float64
    ρ::Float64
    A::AbstractMatrix
    Win::AbstractMatrix
    Wout::AbstractMatrix
    c::AbstractVector
    r::AbstractVector
end
function Base.show(io::IO, s::prct1)
    print(io, "f: R^$(s.M) → R^$(s.P)")
end
function print(s::prct1)
    println("""
    f: R^$(s.M) → R^$(s.P)
      reservoir size N = $(s.N)
       leakage reate α = $(s.α)
    sparsity of Wout β = $(s.β)
     spectral radius ρ = $(s.ρ)
    """)
    return nothing
end
function (s::prct1)(U)    
    encode(r, u) = (1-s.α)*r + tanh.(s.A * r + s.Win * u)
    decode(r, u) = s.Wout * encode(r, u) .+ s.c

    r_ = [s.r]
    [push!(r_, encode(r_[end], u)) for u in eachcol(U)]
    pop!(r_); R = stack(r_)
    S_pred = s.Wout * R .+ s.c
    return S_pred    
end
function reservoir_computing(U::AbstractMatrix, S::AbstractMatrix;
    seed = 0, N = 500, D = 2, α = 1.0, β = 1e-2, ρ = 1.0)

    Random.seed!(seed)
    M = size(U, 1)
    P = size(S, 1)
    RN = erdos_renyi(N, (D*N ÷ 2))
    A = adjacency_matrix(RN) .* 2(rand(N, N) .- .5)
    A = ρ*A / maximum(sqrt.(sum.(abs2, eigen(Matrix(A)).values)))
    Win = sparse(stack([shuffle([1; zeros(M-1)]) for _ in 1:N])') .* 2(rand(N, M) .- .5)
    encode(r, u) = (1-α)*r + tanh.(A * r + Win * u)
    r_ = [randn(N)]
    [push!(r_, encode(r_[end], u)) for u in eachcol(U)]
    pop!(r_); R = stack(r_)
    Wout = S*R'inv(R*R' + (β*LinearAlgebra.I))
    c = -vec(Wout * mean(R, dims = 2) - mean(S, dims = 2))
    return prct1(M, P, N, α, β, ρ, A, Win, Wout, c, r_[end])
end



asdf = reservoir_computing(U, S)

asdf |> print
asdf.r

asdf(U)