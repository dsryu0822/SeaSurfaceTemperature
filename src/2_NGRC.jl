ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,5]])
include("0_utils.jl")
include("1_datacall.jl")

lstm = CSV.read("lstm_optimal_points_prediction_1.csv", DataFrame)
lstm = lstm[lstm.date .< Date(2026), :]

trng = Matrix(nino34[(T .< Date(2024)), :])
test = Matrix(nino34[(Date(2025) .≤ T), :])

candy_observer = sort([1,31301,62601,93901,125201,156501,42,31342,62642,93942,125242,156542,84,31384,62684,93984,125284,156584,126,31426,62726,94026,125326,156626,167,31467,62767,94067,125367,156667,209,31509,62809,94109,125409,156709,251,31551,62851,94151,125451,156751,292,31592,62892,94192,125492,156792,334,31634,62934,94234,125534,156834,376,31676,62976,94276,125576,156876,417,31717,63017,94317,125617,156917,459,31759,63059,94359,125659,156959,501,31801,63101,94401,125701,157001,542,31842,63142,94442,125742,157042,584,31884,63184,94484,125784,157084,626,31926,63226,94526,125826,157126])
id_optimal = [209, 31342, 31509, 31634, 63142, 93901, 94026, 94359, 125367, 125659, 125701, 156709]
id_target = setdiff(axes(nino34, 2), [id_optimal; id_missing])
_id_optimal = findindex(candy_observer, id_optimal)
vrbl = (id_target, id_optimal)
rcv(x) = recover(x, vrbl)

@time f = NGRC(ngrc_sweep(trng, vrbl...;)(100), 100)

"""''''''''''''''''''''''''''''''''''''''''''''''''''

                    overall performance

''''''''''''''''''''''''''''''''''''''''''''''''''"""
lead_ = ["lead$(lpad(d, 3, "0"))" for d in 1:120]
P_i_ = [Matrix(df[:, 5:2:end]) for df in groupby(lstm, :date)]
Q_i_ = [Matrix(df[:, 4:2:end]) for df in groupby(lstm, :date)]
pfmc_m = DataFrame([[] for _ in lead_], lead_)
lwbd_m = DataFrame([[] for _ in lead_], lead_)
@showprogress for k in 1:365
    actl = test[k .+ (1:120), id_target]
    pred = f(P_i_[k])
    fttd = f(Q_i_[k])
    push!(pfmc_m, mean(abs, actl - pred , dims = 2))
    push!(lwbd_m, mean(abs, actl - fttd, dims = 2))
end
pfmc_l = DataFrame([[] for _ in lead_], lead_)
for df = groupby(lstm, :date)
    push!(pfmc_l, mean(abs, Matrix(df[2:end, 4:2:end]) .- Matrix(df[2:end, 5:2:end]), dims = 2))
end



plt_lstm = plot(xticks = [0, 30, 60, 90, 120], xlabel = "lead", ylabel = "MAE", size = [400, 300])
v_ = [[mean(abs, lstm[lstm.lead .== d, j] - lstm[lstm.lead .== d, j+1]) for d in 1:120] for j in 4:2:26]
lat_ = id_optimal / ncol(nino34)
for k in eachindex(v_)[12]
    plot!(plt_lstm, v_[k], color = get(colorschemes[:rainbow], 1 - lat_[k]), lw = 2, alpha = 0.5)
end
plt_lstm

plt_ngrc = plot(xticks = [0, 30, 60, 90, 120], xlabel = "lead", ylabel = "average MAE", size = [400, 300], legend = :topleft, left_margin = 15mm)
μ = mean.(eachcol(pfmc_l))
σ = std.(eachcol(pfmc_l))/10
plot!(plt_ngrc, μ, color = :gray, lw = 2, label = "observer (M points)")
plot!(plt_ngrc, μ .- σ, fillrange = μ .+ σ, fillcolor = :gray, fillalpha = 0.2, label = :none, alpha = 0)
μ = mean.(eachcol(pfmc_m))
σ = std.(eachcol(pfmc_m))/10
plot!(plt_ngrc, μ, color = :orangered, lw = 2, label = "target (P points)")
plot!(plt_ngrc, μ .- σ, fillrange = μ .+ σ, fillcolor = :orangered, fillalpha = 0.2, label = :none, alpha = 0)

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
for k in [31, 121, 251, 311]
    T[T .≥ Date(2025)][k .+ [0, 90]] |> println
end

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


using Plots

# 가짜 데이터 생성 (실제 플롯은 안 보이게)
