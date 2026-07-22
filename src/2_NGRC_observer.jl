ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,5]])
include("0_utils.jl")
include("1_datacall.jl")

# allpoints = collect(Base.product(1:251, 1:626))
candy_observer = [reshape(axes(nino34, 2), 251, 626)[i,j]
    for i in round.(Int64, range(1, 251, length = 6)),
        j in round.(Int64, range(1, 626, length = 16))] |> vec
# candypoints = allpoints[candy_observer]
# plt_hm = heatmap(to3(trng)[:,:,1], framestyle = :none, colorbar = false, ticks = [], size = [500, 100], margin = -1.6mm)
# scatter!(plt_hm, last.(candypoints), first.(candypoints), color = :black, msc = :lightgray, msw = 5)


# candy_observer = sort([1,31301,62601,93901,125201,156501,42,31342,62642,93942,125242,156542,84,31384,62684,93984,125284,156584,126,31426,62726,94026,125326,156626,167,31467,62767,94067,125367,156667,209,31509,62809,94109,125409,156709,251,31551,62851,94151,125451,156751,292,31592,62892,94192,125492,156792,334,31634,62934,94234,125534,156834,376,31676,62976,94276,125576,156876,417,31717,63017,94317,125617,156917,459,31759,63059,94359,125659,156959,501,31801,63101,94401,125701,157001,542,31842,63142,94442,125742,157042,584,31884,63184,94484,125784,157084,626,31926,63226,94526,125826,157126])
trng = Matrix(nino34[(D .< Date(2023)), :])
vldn = Matrix(nino34[(Date(2023) .≤ D .< Date(2024)), :])

begin
    λ_ = exp10.(-3:6)
    result = DataFrame(n_obs = Int64[], bestλ = Float64[], mae = Float64[], id_observer = [])
    local _id_observer = Int64[]
    if isfile("NGRC_sweep_result.csv")
        result = CSV.read("NGRC_sweep_result.csv", DataFrame)
        result.id_observer .= [eval(Meta.parse(id_)) for id_ in result.id_observer]
        _id_observer = result.id_observer[argmin(result.mae)]
    end
    for n_obs = 1:15
        @info "$n_obs phase started"
        rslt = DataFrame(n_obs = zeros(Int64, 96), bestλ = zeros(Float64, 96), mae = zeros(Float64, 96), id_observer = fill([], 96))
        @showprogress for i = eachindex(candy_observer)
            candy = candy_observer[i]
            id_observer = [_id_observer; candy]
            if (length(unique(id_observer)) .!= length(id_observer)) continue end
            id_target = setdiff(axes(nino34, 2), [id_observer; id_missing])
            vrbl = (id_target, id_observer)
            P_i = vldn[:, last(vrbl)]
            P_o = dw(vldn[:, first(vrbl)])

            f_ = ngrc_sweep(trng, vrbl...;)
            mae_ = zeros(length(λ_))
            rmse_ = zeros(length(λ_))
            @threads for k in eachindex(λ_)
                λ = λ_[k]
                f = NGRC(f_(λ), λ)
                fP_i = f(P_i)
                mae_[k] = mae(fP_i, P_o)
            end
            push!(rslt, (n_obs, λ_[argmin(mae_)], minimum(mae_), id_observer))
            CSV.write("NGRC_sweep_result.csv", [result; rslt])
        end
        result = result[.!iszero.(result.n_obs), :]
        CSV.write("NGRC_sweep_result.csv", result)
        presult = result[result.n_obs .== n_obs, :]
        _id_observer = presult.id_observer[argmin(presult.mae)]
    end
end
# result = CSV.read("NGRC_sweep_result.csv", DataFrame)
# result.id_observer .= [eval(Meta.parse(id_)) for id_ in result.id_observer]
# result[argmin(result.mae), :].id_observer
# scatter(result.n_obs, result.mae, shape = :x, color = :black, size = [400,200], xticks = 1:15)

# bestresult = vcat(DataFrame.(first.([sort(df, :mae) for df in groupby(result, :n_obs)]))...)
# round.(100abs.([0; diff(bestresult.mae)]) ./ bestresult.mae, digits = 2)

# candy_observer = [1,31301,62601,93901,125201,156501,42,31342,62642,93942,125242,156542,84,31384,62684,93984,125284,156584,126,31426,62726,94026,125326,156626,167,31467,62767,94067,125367,156667,209,31509,62809,94109,125409,156709,251,31551,62851,94151,125451,156751,292,31592,62892,94192,125492,156792,334,31634,62934,94234,125534,156834,376,31676,62976,94276,125576,156876,417,31717,63017,94317,125617,156917,459,31759,63059,94359,125659,156959,501,31801,63101,94401,125701,157001,542,31842,63142,94442,125742,157042,584,31884,63184,94484,125784,157084,626,31926,63226,94526,125826,157126]
# # candy_observer = sort([1,31301,62601,93901,125201,156501,42,31342,62642,93942,125242,156542,84,31384,62684,93984,125284,156584,126,31426,62726,94026,125326,156626,167,31467,62767,94067,125367,156667,209,31509,62809,94109,125409,156709,251,31551,62851,94151,125451,156751,292,31592,62892,94192,125492,156792,334,31634,62934,94234,125534,156834,376,31676,62976,94276,125576,156876,417,31717,63017,94317,125617,156917,459,31759,63059,94359,125659,156959,501,31801,63101,94401,125701,157001,542,31842,63142,94442,125742,157042,584,31884,63184,94484,125784,157084,626,31926,63226,94526,125826,157126])
# id_optimal = [209, 31342, 31509, 31634, 63142, 93901, 94026, 94359, 125367, 125659, 125701, 156709]


# lat_ = id_optimal / ncol(nino34)
# colors_ = [get(colorschemes[:rainbow], 1 - lat_[k]) for k in eachindex(lat_)]
# sort([1 - lat_[k] for k in eachindex(lat_)])

# allpoints = collect(Base.product(1:626, 1:251))
# candypoints = allpoints[candy_observer]
# plt_hm = heatmap(to3(trng)[:,:,1], framestyle = :none, colorbar = false, ticks = [], size = [500, 100], margin = -1.6mm)
# scatter!(plt_hm, first.(candypoints), last.(candypoints), color = :black, msc = :lightgray, msw = 5)

# # optimalpoints = candypoints[findall(∈(id_optimal), vec(reshape(candy_observer, 6, :)'))]
# optimalpoints = collect(Base.product(1:251, 1:626))[id_optimal]
# scatter!(plt_hm, last.(optimalpoints), first.(optimalpoints), color = colors_, shape = :rect, msc = :white, msw = 10)
# X[last.(optimalpoints)]
# Y[first.(optimalpoints)]

# X[last.(allpoints[result[argmin(result.mae), :].id_observer])]
# Y[first.(allpoints[result[argmin(result.mae), :].id_observer])]





# optimalpoints = collect(Base.product(1:251, 1:626))[candy_observer][findall(∈(id_optimal), candy_observer)]