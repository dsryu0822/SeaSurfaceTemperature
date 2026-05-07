using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :NoiseRobustDifferentiation,
    :Graphs, :Random, :LinearAlgebra, :SparseArrays, :Plots,
    :ProgressMeter] .|> string
required_pkg = filter(p -> isnothing(Base.find_package(p)), unique(packages))
if !isempty(required_pkg)
    Pkg.add(required_pkg)
end
@time eval(Meta.parse("using $(join(unique(packages), ", "))"))
println("All packages loaded")

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
# whereis(ids) = reshape([k ∈ ids for k in 1:150], 10, 15)
whereis(ids) = reshape([k ∈ ids for k in 1:157126], 626, 251)
tensor2dataframe(tnsr) = DataFrame(stack(vec(eachslice(tnsr, dims = (1,2)))), :auto)
dataframe2tensor(df) = reshape(Matrix(df)', 10, 15, :)


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
function eof(X::AbstractMatrix)
    U, Σ, V = svd(X)
    return EOF(U, Σ, V)
end
function (f::EOF)(Û)
    nu = size(Û, 2)
    Ŝ = Diagonal(f.Σ[1:nu])
    V̂ = f.V[:, 1:nu]
    return Û * Ŝ * V̂'
end