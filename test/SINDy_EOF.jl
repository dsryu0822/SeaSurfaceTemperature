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
U, S, V = svd(Matrix(data))
plot(legend = :bottomright)
plot!(T, data.z1, label = "z1")
plot!(T, data.z2, label = "z2")
plot!(T, data.z3, label = "z3")
plot!(T, data.z150, label = "z150")
(S.^2) ./ sum(S .^ 2)
(S) ./ sum(S)
plot((S) ./ sum(S), seriestype=:scatter, title="Scree Plot", xlabel="Mode", ylabel="Explained variance", legend=false, yscale = :log10, color = :black)
svd(Matrix(ddata[:, 1:dim]))
plot(
    plot(T, U[:, 1], color = 1, ylabel = "EOF 1"),
    plot(T, U[:, 2], color = 2, ylabel = "EOF 2"),
    plot(T, U[:, 3], color = 3, ylabel = "EOF 3"),
    layout = (3, 1), legend = :none
)
# plot(data.z1)
V

plot(eof.u2, eof.u3)
eof = DataFrame(u1 = U[:, 1], u2 = U[:, 2], u3 = U[:, 3])
deof = add_diff(eof, method = :TVD)
uvrbl = (names(deof)[1:ncol(eof)], names(deof)[(ncol(eof)+1):end])
ucnfg = cook(last(uvrbl); poly = 0:3)
fu = SINDy(deof, uvrbl, ucnfg)
fu |> print
fu.r2


eof = add_diff(DataFrame(u = U[:, 2]), method = :TVD)
rename!(eof, ["v", "u"])
plot(plot(eof.u, ylabel = "u", color = 1), plot(eof.v, ylabel = "v = du", color = 2), layout = (2, 1), legend = :none, xticks = 365*(0:5))
plt_uv = plot(eof.u, eof.v, ticks = [], xlabel = "u", ylabel = "v", size = [480, 480], color = :gray, label = "traj")
anime = @animate for t = 1:2:(2*365)
    scatter(plt_uv, eof.u[[t]], eof.v[[t]], color = :black, label = "t = $t", msw = 0)
end
gif(anime, "uv.gif", fps = 60)

deof = add_diff(eof, method = :TVD)
uvrbl = (names(deof)[1:ncol(eof)], names(deof)[(ncol(eof)+1):end])
ucnfg = cook(last(uvrbl); poly = 0:2)
fu = SINDy(deof, uvrbl, ucnfg, λ = 1e-6); fu |> print
fu.r2

newVU = solve(fu, [deof[1, 3:4]...], 0:0.01:365)[1:100:end, :]
newV, newU = eachcol(newVU)
plot(U[1:365, 2], color = 1, label = "u(1st EOF)")
plot!(newVU[:, 2], label = "SINDy")

newVU = solve(fu, [deof[1, 3:4]...], 0:0.01:(5*365))[1:100:end, :]
plot(U[1:(5*365), 2], color = 1, label = "u(1st EOF)")
plot!(newVU[:, 2], label = "SINDy")
plot(abs.(U[1:(5*365), 2] - newVU[:, 2][Not(end)]), label = "error", color = :black)

k = 1
Û = U[:, 1:k]
Ŝ = Diagonal(S[1:k])
V̂ = V[:, 1:k]
reconstructed = Û * Ŝ * V̂'
# mean(abs2, reconstructed - Matrix(data))

newZ = newU * Ŝ * V̂'
newtnsr = reshape(newZ, :, 10, 15)
heatmap(newtnsr[1, :, :])
heatmap(tnsr[:, :, 1])

sqrt(mean(abs2, newtnsr[30, :, :] - tnsr[:, :, 30]))

scatter(U[:, 2], xlims = [1, 3*365])
