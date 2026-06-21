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

Random.seed!(0)
id_ = [[[rand(setdiff(1:ncol(data), id_missing),  4) for _ in 1:30];
       [rand(setdiff(1:ncol(data), id_missing), 12) for _ in 1:30];
       ]; [[[1,626,156501,157126]; rand(setdiff(1:ncol(data), id_missing), 8)] for _ in 1:30]]
id_ = sort.(unique.(id_))
@info length.(id_)
title_ = ["rand4_" .* lpad.(1:30, 2, "0");
          "rand12_" .* lpad.(1:30, 2, "0");
          "randmix_" .* lpad.(1:30, 2, "0");
          ]
# sort(unique([id_...;]))
# for k in 1:8
#     x_ = (whereis(id_[k]) .* (1:626))[whereis(id_[k])]
#     y_ = ((1:251)' .* whereis(id_[k]))[whereis(id_[k])]
#     scatter(x_, y_, title = "$k: $(title_[k])", size = [800, 200], xlims = [-10, 636], ylims = [-10, 260], legend = :none, color = :black, xticks = [0, 626], yticks = [0, 251])
#     png("$(k)_$(title_[k]).png")
# end

resultRAND = DataFrame(zeros(1, length(title_)), title_)
optλ_ = []
@async for scene = eachindex(id_)
    id_observer = id_[scene]
    id_target = setdiff(axes(data, 2), [id_missing; id_observer])
    vrbl = (id_target, id_observer)
    viput = data[Year.(T) .== Year(2023), last(vrbl)]
    voput = data[Year.(T) .== Year(2023), first(vrbl)] |> dw |> Matrix

    err_ = []
    λ_ = logrange(1e-4, 1e+5, 10)
    for λ in λ_
        g = ngrc(vtrng, vrbl; λ)
        push!(err_, mae(voput, (g(viput) |> Matrix)))
    end
    push!(optλ_, λ_[argmin(err_)])
    
    @time f = ngrc(trng, vrbl; λ = optλ_[scene])

    # @showprogress for t0 = 1:365
    #     actl = f(test[t0 .+ (0:30), last(vrbl)])
    #     prdt = Matrix(test[t0 .+ (1:30), first(vrbl)])
    #     resultRAND[t0, scene] = mae(actl, prdt)
    # end
    actl = f(test[1 .+ (0:365), last(vrbl)])
    prdt = Matrix(test[1 .+ (1:365), first(vrbl)])
    resultRAND[1, scene] = mae(actl, prdt)
    CSV.write("RAND_scenario.csv", resultRAND)
end
# bar(title_, mean.(eachcol(resultRAND)), legend = :none, color = :white, ylabel = "average MAE over 30 days", title = "actual observer equiped")
# png("RAND_observer.png")
resultRAND = CSV.read("RAND_scenario.csv", DataFrame)

plot(
    histogram(collect(resultRAND[1,  1:30]), bins = 10, xlabel = "MAE", title =  "rand4", legend = :none, color = 1),
    histogram(collect(resultRAND[1, 31:60]), bins = 10, xlabel = "MAE", title = "rand12", legend = :none, color = 2),
    histogram(collect(resultRAND[1, 61:90]), bins = 10, xlabel = "MAE", title = "randmix", legend = :none, color = 3),
    layout = (1, 3), xlims = [0.2, .5], size = [900, 300], margin = 5mm
)
resultNGRC = CSV.read("NGRC_scenario.csv", DataFrame)

comparison = [
    minimum(collect(resultRAND[1,  1:30]))
    mean(resultNGRC[:, 1])
    maximum(collect(resultRAND[1,  1:30]))
    minimum(collect(resultRAND[1, 31:60]))
    mean(resultNGRC[:, 2])
    maximum(collect(resultRAND[1, 31:60]))
    minimum(collect(resultRAND[1, 61:90]))
    maximum(collect(resultRAND[1, 61:90]))
]
title_comparison = ["rand4_best", "corner4", "rand4_worst", "rand12_best", "corner12", "rand12_worst", "randmix_best", "randmix_worst"]

bar(title_comparison, comparison, permute = (:y, :x), ylims = [0, 0.5], legend = :none, ylabel = "MAE", color = [1,1,1,2,2,2,3,3])
