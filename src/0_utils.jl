using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :NoiseRobustDifferentiation,
    :Graphs, :Random, :LinearAlgebra, :SparseArrays, :Plots,
    :ProgressMeter] .|> string
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


"""'''''''''''''''''''''''''''''''''''''''''''''

                reservoir computing

'''''''''''''''''''''''''''''''''''''''''''''"""

struct Reservoir
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
function Base.show(io::IO, s::Reservoir)
    Base.print(io, "f: R^$(s.M) → R^$(s.P)")
end
function Base.print(s::Reservoir)
    Base.print("""
    f: R^$(s.M) → R^$(s.P)
      reservoir size N = $(s.N)
       leakage reate α = $(s.α)
    sparsity of Wout β = $(s.β)
     spectral radius ρ = $(s.ρ)
    """)
    return nothing
end
function (s::Reservoir)(U)    
    encode(r, u) = (1-s.α)*r + s.α*tanh.(s.A * r + s.Win * u)
    decode(r, u) = s.Wout * encode(r, u) .+ s.c

    r_ = [s.r]
    [push!(r_, encode(r_[end], u)) for u in eachcol(U)]
    popfirst!(r_); R = stack(r_)
    S_pred = s.Wout * R .+ s.c
    return S_pred    
end
function reservoir_computing(U::AbstractMatrix, S::AbstractMatrix;
    seed = -1, warmup = 10,
    N = 500, D = 2, α = 0.5, β = 1e-2, ρ = 1.0, σ = 1)

    if seed ≥ 0 Random.seed!(seed) end
    M = size(U, 1)
    P = size(S, 1)
    RN = erdos_renyi(N, (D*N ÷ 2))
    A = adjacency_matrix(RN) .* 2(rand(N, N) .- .5)
    A = ρ*A / maximum(abs.(eigen(Matrix(A)).values))
    Win = σ*(sparse(stack([shuffle([1; zeros(M-1)]) for _ in 1:N])') .* 2(rand(N, M) .- .5))
    encode(r, u) = (1-α)*r + α*tanh.(A * r + Win * u)
    r_ = [randn(N)]
    [push!(r_, encode(r_[end], u)) for u in eachcol(U)]
    popfirst!(r_); R = stack(r_)
    R = R[:, (warmup+1):end]
    S = S[:, (warmup+1):end]
    Wout = S*R'inv(R*R' + (β*LinearAlgebra.I))
    c = -vec(Wout * mean(R, dims = 2) - mean(S, dims = 2))
    return Reservoir(M, P, N, α, β, ρ, A, Win, Wout, c, r_[end])
end
