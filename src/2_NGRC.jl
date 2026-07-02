include("0_utils.jl")
include("1_datacall.jl")
include.(readdir("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core", join=true))


candy_observer = sort([1,31301,62601,93901,125201,156501,42,31342,62642,93942,125242,156542,84,31384,62684,93984,125284,156584,126,31426,62726,94026,125326,156626,167,31467,62767,94067,125367,156667,209,31509,62809,94109,125409,156709,251,31551,62851,94151,125451,156751,292,31592,62892,94192,125492,156792,334,31634,62934,94234,125534,156834,376,31676,62976,94276,125576,156876,417,31717,63017,94317,125617,156917,459,31759,63059,94359,125659,156959,501,31801,63101,94401,125701,157001,542,31842,63142,94442,125742,157042,584,31884,63184,94484,125784,157084,626,31926,63226,94526,125826,157126])
trng = Matrix(nino34[(T .< Date(2023)), :])
vldn = Matrix(nino34[(Date(2023) .≤ T .< Date(2024)), :])

begin
    λ_ = exp10.(-6:3)
    result = DataFrame(n_obs = Int64[], bestλ = Float64[], mae = Float64[], id_observer = [])
    if isfile("NGRC_sweep_result.csv") result = CSV.read("NGRC_sweep_result.csv", DataFrame) end
    _id_observer = Int64[]
    for n_obs = 1:12
        @info "$n_obs phase started"
        @showprogress @threads for candy = candy_observer
            id_observer = [_id_observer; candy]
            if any(Ref(id_observer) .== result.id_observer) continue end
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
            push!(result, (n_obs, λ_[argmin(mae_)], minimum(mae_), id_observer))
            # CSV.write("NGRC_sweep_result.csv", result)
        end
        presult = result[result.n_obs .== n_obs, :]
        global _id_observer = presult.id_observer[argmin(presult.mae)]
    end
end
result = CSV.read("NGRC_sweep_result.csv", DataFrame)
result[argmin(result.mae), :]

scatter(result.n_obs, result.mae, xlabel = L"M", ylabel = "validation MAE", xticks = 1:12, alpha = .5, size = [300, 200], color = :black, shape = :x, ylims = [.2, .9])
png("temp")

lstm = CSV.read("lstm_optimal_points_prediction_1.csv", DataFrame)
lstm = lstm[lstm.date .< Date(2026), :]
_trng = Matrix(nino34[(T .< Date(2024)), :])
test = Matrix(nino34[(Date(2025) .≤ T), :])

id_optimal = [209, 31342, 31509, 31634, 63142, 93901, 94026, 94359, 125367, 125659, 125701, 156709]
id_target = setdiff(axes(nino34, 2), [id_optimal; id_missing])
_id_optimal = findindex(candy_observer, id_optimal)
vrbl = (id_target, id_optimal)
rcv(x) = recover(x, vrbl)

# ngrc_ = ngrc_sweep(_trng, vrbl...;)
# f = ngrc(_trng, vrbl...; λ = 100)
@time f = NGRC(ngrc_sweep(trng, vrbl...;)(λ), λ)


P_i_ = [Matrix(df[:, 5:2:end]) for df in groupby(lstm, :date)]
Q_i_ = [Matrix(df[:, 4:2:end]) for df in groupby(lstm, :date)]
lead_ = ["lead$(lpad(d, 3, "0"))" for d in 1:120]
pfmc = DataFrame([[] for _ in lead_], lead_)
# pfmc_rmse = DataFrame([[] for _ in lead_], lead_)
lwbd = DataFrame([[] for _ in lead_], lead_)
@showprogress for k in 1:365
    actl = test[k .+ (1:120), id_target]
    pred = f(P_i_[k])
    fttd = f(Q_i_[k])
    push!(pfmc, mean(abs, actl - pred , dims = 2))
    # push!(pfmc_rmse, sqrt.(mean(abs2, actl - pred, dims = 2)))
    push!(lwbd, mean(abs, actl - fttd, dims = 2))
end

T[T .≥ Date(2025)]
rmse(actl, pred, dims = 2)
CSV.write("final_performance_mae.csv", [DataFrame(startdate = Date(2025,1,1):Date(2025,12,31)) pfmc])
CSV.write("final_performance_rmse.csv", [DataFrame(startdate = Date(2025,1,1):Date(2025,12,31)) pfmc_rmse])

plt_lstm = plot(xticks = [0, 30, 60, 90, 120], xlabel = "lead", ylabel = "MAE", size = [400, 300])
v_ = [[mean(abs, lstm[lstm.lead .== d, j] - lstm[lstm.lead .== d, j+1]) for d in 1:120] for j in 4:2:26]
for k in eachindex(v_)
    plot!(plt_lstm, v_[k], color = k)
end
plt_ngrc = plot(xticks = [0, 30, 60, 90, 120], xlabel = "lead", ylabel = "average MAE", size = [400, 300], legend = true)
plot!(plt_ngrc, [mean(abs, lstm[lstm.lead .== d, 4:2:end] - lstm[lstm.lead .== d, 5:2:end]) for d in 1:120], color = :black, lw = 2, label = "observer (M points)")
plot!(plt_ngrc, mean.(eachcol(pfmc)), color = :red, lw = 2, label = "target (P points)")
plot(
    plt_lstm, plt_ngrc,
    size = [800, 300], margin = 3mm
)

for k = 1:5:365
plot(
    heatmap(to2(test[90+k, :])),
    heatmap(to2(rcv(f(P_i_[k])[90,:]))),
    layout = (:, 1), clims = (24, 30), ticks = [], colorbar = false, margin = -1.6mm, size = [500, 200]
)
png("k=$k")
end
T[T .≥ Date(2025)][1 .+ [0, 90]]
T[T .≥ Date(2025)][121 .+ [0, 90]]
T[T .≥ Date(2025)][244 .+ [0, 90]]
T[T .≥ Date(2025)][365 .+ [0, 90]]

T365 = T[Date(2025) .≤ T]
T365[Day.(T365) .== Day(1)]
for k in findall(Day.(T365) .== Day(1))[1:12]
    Z = to2(test[120+k, :]) - to2(rcv(f(P_i_[k])[120,:]))
    plt_err_spc = heatmap(Z, ticks = [], colorbar = false, margin = -1.6mm, color = :balance)
    scatter!(plt_err_spc, first.(optimalpoints), last.(optimalpoints), color = :black, msc = :white, shape = :rect)
    plt_abserr_ = heatmap(abs.(Z), ticks = [], colorbar = false, margin = -1.6mm, color = cgrad(:RdYlGn, rev = true))
    scatter!(plt_abserr_, first.(optimalpoints), last.(optimalpoints), color = :black, msc = :white, shape = :rect)
    plot(plt_err_spc, plt_abserr_, layout = (:, 1), size = [500, 400], plot_title = "$(T365[k]) → $(T365[k+120])")
    png("error_$(T365[k])")
end


allpoints = collect(Base.product(1:626, 1:251))
candypoints = allpoints[candy_observer]
optimalpoints = allpoints[id_optimal]
plt_hm = heatmap(to3(test)[:,:,1], framestyle = :none, colorbar = false, ticks = [], size = [500, 100], margin = -1.6mm)
scatter!(plt_hm, first.(candypoints), last.(candypoints), color = :black, msc = :white)
scatter!(plt_hm, first.(optimalpoints), last.(optimalpoints), color = 3, msc = :black, shape = :rect)
X[last.(optimalpoints)]
Y[first.(optimalpoints)]