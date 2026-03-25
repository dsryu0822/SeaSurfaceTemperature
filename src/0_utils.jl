using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :NoiseRobustDifferentiation,
    :Graphs, :Random, :LinearAlgebra, :SparseArrays, :Plots,
    :ProgressMeter] .|> string
required = filter(p -> isnothing(Base.find_package(p)), unique(packages))
if !isempty(required)
    Pkg.add(required)
end
@time eval(Meta.parse("using $(join(unique(packages), ", "))"))
println("All packages loaded")

tensor2dataframe(tnsr) = DataFrame(stack(vec(eachslice(tnsr, dims = (1,2)))), :auto)
dataframe2tensor(df) = reshape(Matrix(df)', 10, 15, :)
mae(x, y; kargs...) = mean(abs, x - y; kargs...)
rmse(x, y; kargs...) = sqrt.(mean(abs2, x - y; kargs...))
function add_diff(D::AbstractDataFrame; method = :FDM)
    dnames = "d" .* names(D)
    if method == :FDM
        return [DataFrame(diff(Matrix(D), dims = 1), dnames) D[1:(end-1), :]]
    elseif method == :TVD
        return [DataFrame([tvdiff(z, 10, 100, dx = 1) for z in eachcol(D)], dnames) D]
    else
        throw(ArgumentError("method must be :FDM or :TVD"))
    end
end
function variablenames(df)
    names_df = names(df)
    if mod(length(names_df), 2) == 0
        return (names_df[1:(end÷2)], names_df[(end÷2 + 1):end])
    else
        throw(ArgumentError("DataFrame must have an even number of columns"))
    end
end
function frequency(inventory)
    freq = Dict{eltype(inventory), Int64}()
    for item in inventory
        freq[item] = get(freq, item, 0) + 1
    end
    return freq
end
id_flip(ids) = setdiff(1:150, ids)
whereis(ids) = reshape([k ∈ ids for k in 1:150], 10, 15)


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
    ξ::Float64
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
    encode(r, u) = (1-s.α)*r + s.α*tanh.(s.A * r + s.Win * u .+ s.ξ)

    r_ = [s.r]
    [push!(r_, encode(r_[end], u)) for u in eachcol(U)]
    popfirst!(r_); R = stack(r_)
    S_pred = s.Wout * R .+ s.c
    return S_pred    
end
function reservoir_computing(U::AbstractMatrix, S::AbstractMatrix;
    seed = -1, warmup = 10,
    N = 500, D = 2, α = 0.5, β = 1e-2, ρ = 1.0, σ = 1.0, ξ = 1.0)

    if seed ≥ 0 Random.seed!(seed) end
    M = size(U, 1)
    P = size(S, 1)
    RN = erdos_renyi(N, (D*N ÷ 2))
    A = adjacency_matrix(RN) .* 2(rand(N, N) .- .5)
    A = ρ*A / maximum(abs.(eigen(Matrix(A)).values))
    Win = σ*(sparse(stack([shuffle([1; zeros(M-1)]) for _ in 1:N])') .* 2(rand(N, M) .- .5))
    encode(r, u) = (1-α)*r + α*tanh.(A * r + Win * u .+ ξ)
    r_ = [zeros(N)]
    [push!(r_, encode(r_[end], u)) for u in eachcol(U)]
    popfirst!(r_); R = stack(r_)
    R = R[:, (warmup+1):end]
    S = S[:, (warmup+1):end]
    Wout = S*R'inv(R*R' + (β*LinearAlgebra.I))
    c = -vec(Wout * mean(R, dims = 2) - mean(S, dims = 2))
    return Reservoir(M, P, N, α, β, ρ, A, ξ, Win, Wout, c, r_[end])
end




"""'''''''''''''''''''''''''''''''''''''''''''''

        Empirical orthogonal function (EOF)

'''''''''''''''''''''''''''''''''''''''''''''"""
struct EOF
    U::AbstractMatrix
    Σ::AbstractVector
    V::AbstractMatrix
end
function Base.show(io::IO, s::EOF)
    Base.print(io, "EOF for R^$(size(s.U, 1)) × R^$(size(s.V, 1))")
end
function Base.print(s::EOF)
    Base.print("""
    EOF for R^$(size(s.U, 1)) × R^$(size(s.V, 1))
      singular values Σ = $(s.Σ)
    """)
    return nothing
end
function EOF(X::AbstractMatrix)
    U, Σ, V = svd(X)
    return EOF(U, Σ, V)
end
function (f::EOF)(Û)
    nu = size(Û, 2)
    Ŝ = Diagonal(f.Σ[1:nu])
    V̂ = f.V[:, 1:nu]
    return Û * Ŝ * V̂'
end