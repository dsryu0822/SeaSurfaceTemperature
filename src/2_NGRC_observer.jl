ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,5]])
include("0_utils.jl")
include("1_datacall.jl")

candy_observer = [reshape(axes(nino34, 2), 251, 626)[i,j]
    for i in round.(Int64, range(1, 251, length = 6)),
        j in round.(Int64, range(1, 626, length = 16))] |> vec

trng = Float32.(Matrix(nino34[(D .< Date(2024)), Not(id_missing)]))
vldn = Float32.(Matrix(nino34[(Date(2024) .≤ D .< Date(2025)), Not(id_missing)]))
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
# result = CSV.read("NGRC_sweep_result.csv", DataFrame)
# result.id_observer .= [eval(Meta.parse(id_)) for id_ in result.id_observer]
# result[argmin(result.mae), :].id_observer
# scatter(result.n_obs, result.mae, shape = :x, color = :black, size = [400,200], xticks = 1:15)

# bestresult = vcat(DataFrame.(first.([sort(df, :mae) for df in groupby(result, :n_obs)]))...)
# id_optimal = bestresult.id_observer[argmin(bestresult.mae)]

# round.(100abs.([0; diff(bestresult.mae)]) ./ bestresult.mae, digits = 2)
# X[first.(allpoints[last.(bestresult.id_observer)])]
# Y[last.(allpoints[last.(bestresult.id_observer)])]



# allpoints = collect(Base.product(1:251, 1:626))
# optimalpoints = allpoints[id_optimal]
# candypoints = allpoints[candy_observer]

# RdYlBu_r = cgrad(:RdYlBu, rev = true)
# vanimo_r = cgrad(:vanimo, rev = true)
# colors_ = [get(vanimo_r, lat) for lat in (id_optimal / ncol(nino34))]
# plt_hm = heatmap(to3(trng)[:,:,1], framestyle = :none, color = RdYlBu_r, colorbar = false, ticks = [], size = [500, 100], margin = -1.6mm)
# scatter(plt_hm, last.(candypoints), first.(candypoints), color = :black, msc = :white, msw = 10)
# scatter(plt_hm, last.(optimalpoints), first.(optimalpoints), color = colors_, shape = :rect, msc = :white, msw = 10)

