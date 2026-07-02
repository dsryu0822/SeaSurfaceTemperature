# if Sys.KERNEL == :NT
#     ROOT = "C:/Users/rmsms/OneDrive/lab/"
# else
#     ROOT = "/home/chaos"
# end

# using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
# import Pkg
# packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
#     :LinearAlgebra, :Distances, :JLD2, :NoiseRobustDifferentiation,
#     :Graphs, :Random, :LinearAlgebra, :SparseArrays, :Plots,
#     :ProgressMeter] .|> string
# required_pkg = filter(p -> isnothing(Base.find_package(p)), unique(packages))
# if !isempty(required_pkg)
#     Pkg.add(required_pkg)
# end
# @time "📚 packages loaded" eval(Meta.parse("using $(join(unique(packages), ", "))"))

import Base.-
-(a::AbstractDataFrame, b::AbstractDataFrame) = Matrix(a) - Matrix(b)
-(a::AbstractDataFrame, b::AbstractMatrix) = Matrix(a) - b
-(a::AbstractMatrix, b::AbstractDataFrame) = a - Matrix(b)

function variablenames(df)
    names_df = names(df)
    if mod(length(names_df), 2) == 0
        return (names_df[1:(end÷2)], names_df[(end÷2 + 1):end])
    else
        throw(ArgumentError("DataFrame must have an even number of columns"))
    end
end
tensor2dataframe(tnsr) = DataFrame(stack(vec(eachslice(tnsr, dims = (1,2)))), :auto)
dataframe2tensor(df) = reshape(Matrix(df)', 10, 15, :)

# recover(matrices, indices) = matrices[:, sortperm(indices)]
# function recoverH(matrices, indice = [])
#     matrices = Matrix(matrices)
#     void = fill(missing, size(matrices, 1), length(id_missing))
#     return recover([matrices;; void], [indice; id_missing])
# end


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


to3(array2) = reshape(Matrix(array2)', length(X), length(Y), :)
to2(array1) = reshape(array1, length(X), length(Y))
findindex(big, small) = [findfirst(==(x), big) for x in small]
function recover(vector, variables)
    filled = Array{Union{Float64, Missing}}(missing, ncol(nino34))
    filled[first(variables)] .= vector
    return filled
end