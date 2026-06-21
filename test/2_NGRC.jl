include("0_utils.jl")
# JLD2.@save "G:/seasurface/nino3.4/data.jld2" data T X Y
@time JLD2.@load "G:/seasurface/nino3.4/data.jld2"
include.(readdir("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core", join=true))

to2(x) = reshape(collect(x), length(Y), length(X))
datasplit(array, bits) = (array[bits,:], array[.!bits,:])
recover(matrices, indices) = matrices[:, sortperm(indices)]

id_missing = findall((eltype.(eachcol(data))) .== Missing)
function recoverH(matrices, indice)
    void = fill(missing, size(matrices, 1), length(id_missing))
    return recover([matrices;; void], [indice; id_missing])
end

id_observer = [1, 626, 156501, 157126]
id_observer = [1:125:626; 156501:125:157126]
id_target = setdiff(axes(data, 2), [id_missing; id_observer])
vrbl = (id_target, id_observer)


# heatmap(Y, X, to2(data[1, :])', size = (800, 200), yticks = [-5, 0, 5])
trng, _ = datasplit(data, T .< Date(2023, 1, 1))
days(day, Δ = 30) = T .∈ Ref(day:Day(1):(day+Day(Δ)))
trng[:, id_observer]
@time f = ngrc(trng, vrbl; λ = 1e+3)


vtrng = data[Year.(T) .≤ Year(2022), :]
vinput = data[Year.(T) .== Year(2023), last(vrbl)]
voutput = data[Year.(T) .== Year(2023), first(vrbl)] |> dw |> Matrix
err_ = []
λ_ = logrange(1e-10, 1e+5, 16)
for λ in λ_
    g = ngrc(vtrng, vrbl; λ)
    push!(err_, rmse(voutput, (g(vinput) |> Matrix)))
end
plot(λ_, err_, xscale = :log10, xticks = λ_[1:2:end], color = :black, legend = :none)


bit_period = days(Date(2023, 08, 31), 91)
osrv = dw(data[bit_period, last(vrbl)]) |> Matrix
prdt = f((1 .+ 0.01rand([-1, 1], 92, 4)) .* Matrix(data[bit_period, last(vrbl)]))
@time actl = dw(data[bit_period, first(vrbl)]) |> Matrix
sqrt(mean(abs2, (actl - prdt)))

rcvdA = recoverH([actl;; osrv], [id_target; id_observer])
rcvdP = recoverH([prdt;; osrv], [id_target; id_observer])
plot(
    heatmap(Y, X, to2(rcvdA[11, :])'),
    heatmap(Y, X, to2(rcvdP[11, :])'),
    clims = extrema(skipmissing([rcvdA; rcvdP])),
    layout = (2, 1)
)


"""''''''''''''''''''''''''''''''''''''

            LSTM observer

''''''''''''''''''''''''''''''''''''"""

obsv = CSV.read("lstm_prediction30.csv", DataFrame)
# names(obsv)[[4, 24, 6, 26]]
# names(obsv)[[4:4:24; 6:4:26]]

lead30 = [df[:, [4, 24, 6, 26]] for df in groupby(obsv, :date)]
lead30 = [df[:, [4:4:24; 6:4:26]] for df in groupby(obsv, :date)]
# read30 = [data[t0 .+ (1:30), id_observer] for t0 in findall(Date(2023, 1, 1) .≤ T .≤ Date(2023, 12, 31))]

rmseover30 = []
rmseat30 = []
days365 = findall(Date(2023, 1, 1) .≤ T .≤ Date(2023, 12, 31))
for tk in eachindex(days365)
    actl = Matrix(data[days365[tk] .+ (1:30), id_target])
    # prdt = f(lead30[tk])
    prdt = f(data[days365[tk] .+ (0:30), id_observer])
    push!(rmseover30, rmse(actl, prdt))
    push!(rmseat30, rmse(actl[end, :], prdt[end, :]))
end

sargs = (; legend = :none, color = :black, ylims = [0, 1.3], ylabel = "RMSE")
scatter(T[1462:1826], rmseover30; sargs...)
mean(rmseover30)
scatter(T[1462:1826], rmseat30; sargs...)
mean(rmseat30)


[names(obsv[:, [4:4:24; 6:4:26]]) names(data[:, id_observer])]
Random.seed!(42)
id_ = [
    [1, 626, 156501, 157126],
    [1:125:626; 156501:125:157126],
    [72293, 72313, 84813, 84833],
    sort(setdiff([id_missing .- 1; id_missing .+ 1], id_missing)),
    1:(626*25):157126,
    626:(626*25):157126,
    rand(1:157126, 4),
    rand(1:157126, 12),
]
title_ = [
    "corner4",
    "corner12",
    "center",
    "island",
    "east",
    "west",
    "rand4",
    "rand12",
]
    
for k in 1:8
    x_ = (whereis(id_[k]) .* (1:626))[whereis(id_[k])]
    y_ = ((1:251)' .* whereis(id_[k]))[whereis(id_[k])]
    scatter(x_, y_, title = "$k: $(title_[k])", size = [800, 200], xlims = [-10, 636], ylims = [-10, 260], legend = :none, color = :black, xticks = [0, 626], yticks = [0, 251])
    png("$(k)_$(title_[k]).png")
end

sort(unique([id_...;]))