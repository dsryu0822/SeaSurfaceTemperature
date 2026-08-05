ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,5]])
include("0_utils.jl")
# include("1_datacall.jl")

candy_observer = [reshape(axes(nino34, 2), 251, 626)[i,j]
    for i in round.(Int64, range(1, 251, length = 6)),
        j in round.(Int64, range(1, 626, length = 16))] |> vec
allpoints = collect(Base.product(1:251, 1:626))
CSV.write("observer96.csv", [D nino34[:, candy_observer]])

mapcols!(nino34) do col
    eltype(col) <: Union{Float64, Missing} ? passmissing(Float32).(col) : col
end
trng = Matrix(nino34[(D .< Date(2023)), :])
vldn = Matrix(nino34[(Date(2023) .≤ D .< Date(2024)), :])
function λ_sweep()
    λ_ = exp10.(-3:6)
    result = DataFrame(n_obs = Int64[], bestλ = Float64[], mae = Float64[], id_observer = [])
    _id_observer = Int64[]
    if isfile("NGRC_sweep_result.csv")
        result = CSV.read("NGRC_sweep_result.csv", DataFrame)
        result.id_observer .= [eval(Meta.parse(id_)) for id_ in result.id_observer]
        _id_observer = result.id_observer[argmin(result.mae)]
    end
    for n_obs = 1:15
        @info "$n_obs phase started"
        rslt = DataFrame(n_obs = zeros(Int64, 96), bestλ = zeros(Float64, 96), mae = zeros(Float64, 96), id_observer = fill([], 96))
        @showprogress @threads for i = eachindex(candy_observer)
            candy = candy_observer[i]
            id_observer = [_id_observer; candy]
            if (length(unique(id_observer)) .!= length(id_observer)) continue end
            id_target = setdiff(axes(nino34, 2), [id_observer; id_missing])
            vrbl = (id_target, id_observer)
            P_i = vldn[:, last(vrbl)]
            P_o = dw(vldn[:, first(vrbl)])

            f_ = ngrc_sweep(trng, vrbl...;)
            mae_ = zeros(length(λ_))
            for k in eachindex(λ_)
                λ = λ_[k]
                f = NGRC(f_(λ), λ)
                mae_[k] = mae(f(P_i), P_o)
            end
            rslt[i, :] = [n_obs, λ_[argmin(mae_)], minimum(mae_), id_observer]
            CSV.write("NGRC_sweep_result.csv", [result; rslt])
        end
        result = [result; rslt]
        result = result[.!iszero.(result.n_obs), :]
        CSV.write("NGRC_sweep_result.csv", result)
        presult = result[result.n_obs .== n_obs, :]
        _id_observer = presult.id_observer[argmin(presult.mae)]
    end
end
λ_sweep()

"""''''''''''''''''''''''''''''''''''''''''''''''''''

                          analysis

''''''''''''''''''''''''''''''''''''''''''''''''''"""
result = CSV.read("NGRC_sweep_result 2024.csv", DataFrame)
result.id_observer .= [eval(Meta.parse(id_)) for id_ in result.id_observer]
result[argmin(result.mae), :].id_observer
bestresult = vcat(DataFrame.(first.([sort(df, :mae) for df in groupby(result, :n_obs)]))...)

Δ = round.(100abs.([0; diff(bestresult.mae)]) ./ bestresult.mae, digits = 2)
lon = X[first.(allpoints[last.(bestresult.id_observer)])]
lat = Y[last.(allpoints[last.(bestresult.id_observer)])]

_Q(M) = (2M+1)*(M+1)
plt_mae = scatter(result.n_obs, result.mae, shape = :x, color = :black, size = [400,200], xticks = 1:15, ylims = [0.2, 1], xformatter = _ -> "", bottom_margin = -3mm, yticks = [0.2, 1.0])
plt_mae = scatter(_Q.(result.n_obs), result.mae, shape = :x, color = :black)

plt_delta = plot(size = [400,200], xticks = 1:15, ylims = [0, 16], yticks = [0, 16])
plot!(plt_delta, 2:15, Δ[2:15], result.mae, color = :black)
scatter!(plt_delta, [8], [Δ[8]], color = 1, msc = :white, shape = :rect, ms = 4)

plot(plt_mae, plt_delta, layout = (2,1), size = [400, 200], xlims = [0.5, 15.5])
id_optimal = bestresult.id_observer[8]

png("temp")

plt_mae = scatter(result.n_obs, result.mae, shape = :x, color = :black, size = [400,200], xticks = 1:15, ylims = [0.2, 1])
plot!(twinx(), 2:15, Δ[2:15])
scatter!(twinx(), [8], [Δ[8]], yticks = [])


candypoints = allpoints[candy_observer]
optimalpoints = allpoints[id_optimal]

colors_ = [get(vanimo_r, lat) for lat in (id_optimal / ncol(nino34))]
plt_hm = heatmap(to3(trng)[:,:,1], framestyle = :none, color = RdYlBu_r, colorbar = false, ticks = [], size = [500, 100], margin = -1.6mm)
scatter(plt_hm, last.(candypoints), first.(candypoints), color = :black, msc = :white, msw = 10)
scatter(plt_hm, last.(optimalpoints), first.(optimalpoints), color = colors_, shape = :rect, msc = :white, msw = 10)
# for t in 1:120
#     plt_hm = heatmap(to3(trng)[:,:,t], framestyle = :none, color = RdYlBu_r, colorbar = false, ticks = [], size = [500, 100], margin = -1.6mm)
#     png("E:/$t.png")
# end
plt_ts = plot(xticks = 2019:2027, xlims = [2019, 2026.4], size = [900, 300])
for (jk, j) in enumerate(id_optimal)
    plot!(plt_ts, 2019 .+ (1:nrow(nino34))/365, nino34[:, j], color = colors_[jk], alpha = 0.5, lw = 2)
end
plt_ts

