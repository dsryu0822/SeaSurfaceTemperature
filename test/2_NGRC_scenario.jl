include("0_utils.jl")
# JLD2.@save "G:/seasurface/nino3.4/data.jld2" data T X Y
@time JLD2.@load "G:/seasurface/nino3.4/data.jld2"
include.(readdir("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core", join=true))

days(day, Δ = 30) = T .∈ Ref(day:Day(1):(day+Day(Δ)))
to2(x) = reshape(collect(x), length(Y), length(X))
datasplit(array, bits) = (array[bits,:], array[.!bits,:])
recover(matrices, indices) = matrices[:, sortperm(indices)]

id_missing = findall((eltype.(eachcol(data))) .== Missing)
function recoverH(matrices, indice)
    void = fill(missing, size(matrices, 1), length(id_missing))
    return recover([matrices;; void], [indice; id_missing])
end

 trng,  test = datasplit(data, T .< Date(2023, 1, 1))
vtrng, vtest = datasplit(data, T .< Date(2022, 1, 1))

id_ = [
    [1, 626, 156501, 157126],
    [1:125:626; 156501:125:157126],
    [72293, 72313, 84813, 84833],
    sort(setdiff([id_missing .- 1; id_missing .+ 1], id_missing)),
    1:(626*25):157126,
    626:(626*25):157126,
    [70760, 75014, 98887, 110480], # Random.seed!(42); rand(1:157126, 4),
    [14145, 26067, 42090, 47039, 53839, 71812, 81057, 96394, 100472, 103883, 105014, 105801],# rand(1:157126, 12),
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
sort(unique([id_...;]))
# for k in 1:8
#     x_ = (whereis(id_[k]) .* (1:626))[whereis(id_[k])]
#     y_ = ((1:251)' .* whereis(id_[k]))[whereis(id_[k])]
#     scatter(x_, y_, title = "$k: $(title_[k])", size = [800, 200], xlims = [-10, 636], ylims = [-10, 260], legend = :none, color = :black, xticks = [0, 626], yticks = [0, 251])
#     png("$(k)_$(title_[k]).png")
# end

# optλ_ = []
# for scene = eachindex(id_)
#     id_observer = id_[scene]
#     id_target = setdiff(axes(data, 2), [id_missing; id_observer])
#     vrbl = (id_target, id_observer)
#     viput = data[Year.(T) .== Year(2023), last(vrbl)]
#     voput = data[Year.(T) .== Year(2023), first(vrbl)] |> dw |> Matrix

#     err_ = []
#     λ_ = logrange(1e-4, 1e+5, 10)
#     for λ in λ_
#         g = ngrc(vtrng, vrbl; λ)
#         push!(err_, mae(voput, (g(viput) |> Matrix)))
#     end
#     push!(optλ_, λ_[argmin(err_)])
#     # plot(λ_, err_, xscale = :log10, xticks = λ_[1:2:end], color = :black, legend = :none)
#     # png("$(scene)_$(title_[scene])_λ.png")
# end
λ_ = exp10.([2, 3, 3, -2, 2, 4, 0, 2])

resultNGRC = DataFrame(zeros(365, length(title_)), title_)
@async for scene = eachindex(title_)
    id_observer = id_[scene]
    id_target = setdiff(axes(data, 2), [id_missing; id_observer])
    vrbl = (id_target, id_observer)
    @time f = ngrc(trng, vrbl; λ = λ_[scene])

    @showprogress for t0 = 1:365
        actl = Matrix(test[t0 .+ (1:30), first(vrbl)])
        prdt = f(test[t0 .+ (0:30), last(vrbl)])
        resultNGRC[t0, scene] = mae(actl, prdt)
    end
end
# CSV.write("NGRC_scenario.csv", resultNGRC)
# bar(title_, mean.(eachcol(resultNGRC)), legend = :none, color = :white, ylabel = "average MAE over 30 days", title = "actual observer equiped")
# png("NGRC_observer.png")


obsv = CSV.read("lstm_multipoint_prediction_2023.csv", DataFrame)
mae(Matrix(obsv[obsv.lead .== 29, 3:2:end]), Matrix(obsv[obsv.lead .== 29, 4:2:end]))
mae(Matrix(obsv[:, 3:2:end]), Matrix(obsv[:, 4:2:end]))

resultNGRC = DataFrame(zeros(365, length(title_)), title_)
@async for scene = eachindex(title_)
    id_observer = id_[scene]
    id_target = setdiff(axes(data, 2), [id_missing; id_observer])
    vrbl = (id_target, id_observer)
    @time f = ngrc(trng, vrbl; λ = λ_[scene])

    _obsv = [obsv[:, [1]] obsv[:, 4:2:end][:, findall(sort(unique([id_...;])) .∈ Ref(id_[scene]))]]
    _test = [df[:, 2:end] for df in groupby(_obsv, :date)]
    @showprogress for t0 = 1:365
        actl = Matrix(test[t0 .+ (1:29), first(vrbl)])
        prdt = f(_test[t0])
        resultNGRC[t0, scene] = mae(actl, prdt)
    end
end
CSV.write("NGRC_scenario_observer.csv", resultNGRC)
bar(title_, mean.(eachcol(resultNGRC)), legend = :none, color = :white, ylabel = "average MAE over 29 days", title = "LSTM observer equiped")

