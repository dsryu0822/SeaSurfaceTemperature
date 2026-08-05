ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,5]])
include("0_utils.jl")
include("1_datacall.jl")

# id_optimal = [104718, 10693, 146434, 20834, 125601, 146484, 125501, 83734, 125651, 104818, 73493, 136243, 42018, 94326] |> sort
id_optimal = [104718, 10693, 146434, 20834, 125601, 146484, 125501, 83734] |> sort
lstm = CSV.read("lstm_optimal_points_prediction_260727(1).csv", DataFrame)
lstm = lstm[lstm.date .< Date(2026), :]

trng = Matrix(nino34[(D .< Date(2024)), :])
test = Matrix(nino34[(Date(2025) .≤ D), :])

id_target = setdiff(axes(nino34, 2), [id_optimal; id_missing])
vrbl = (id_target, id_optimal)
rcv(x) = recover(x, vrbl)

@time f_ = ngrc_sweep(trng, vrbl...)
@time f = NGRC(f_(1e+2), 1e+2)
"""''''''''''''''''''''''''''''''''''''''''''''''''''

                    overall performance

''''''''''''''''''''''''''''''''''''''''''''''''''"""
lead_ = ["lead$(lpad(d, 3, "0"))" for d in 1:120]
P_i_ = [Matrix(df[:, 5:2:end]) for df in groupby(lstm, :date)]
Q_i_ = [Matrix(df[:, 4:2:end]) for df in groupby(lstm, :date)]
pfmc_m = DataFrame([[] for _ in lead_], lead_)
# lwbd_m = DataFrame([[] for _ in lead_], lead_)
@showprogress for k in 1:365
    actl = test[k .+ (1:120), id_target]
    pred = f(P_i_[k])
    # fttd = f(Q_i_[k])
    push!(pfmc_m, mean(abs, actl - pred , dims = 2))
    # push!(pfmc_m, mean(abs, actl - fttd, dims = 2))
end
pfmc_l = DataFrame([[] for _ in lead_], lead_)
for df = groupby(lstm, :date)
    push!(pfmc_l, mean(abs, Matrix(df[2:end, 4:2:end]) .- Matrix(df[2:end, 5:2:end]), dims = 2))
end



plt_lstm = plot(xticks = [0, 30, 60, 90, 120], yticks = [0.2, 0.6, 1.0], ylims = [0.2, 1.0])
v_ = [[mean(abs, lstm[lstm.lead .== d, j] - lstm[lstm.lead .== d, j+1]) for d in 1:120] for j in 4:2:18]
colors_ = [get(vanimo_r, lat) for lat in (id_optimal / ncol(nino34))]
for k in eachindex(v_)
    plot!(plt_lstm, v_[k], color = colors_[k], lw = 2, alpha = 0.7)
end
plt_lstm

plt_ngrc = plot(xticks = [0, 30, 60, 90, 120], yticks = [0.3, 0.5, 0.7], ylims = [0.3, 0.7], legend = :topright)
μ = mean.(eachcol(pfmc_l))
σ = std.(eachcol(pfmc_l))/5
plot!(plt_ngrc, μ, color = :gray, lw = 2, label = "observer")
plot!(plt_ngrc, μ .- σ, fillrange = μ .+ σ, fillcolor = :gray, fillalpha = 0.2, label = :none, alpha = 0)
μ = mean.(eachcol(pfmc_m))
σ = std.(eachcol(pfmc_m))/5
plot!(plt_ngrc, μ, color = :orangered, lw = 2, label = "target")
plot!(plt_ngrc, μ .- σ, fillrange = μ .+ σ, fillcolor = :orangered, fillalpha = 0.2, label = :none, alpha = 0)

plot(
    plt_lstm, plt_ngrc,
    layout = (:, 1), size = [300, 300]
)


minecraft = CSV.read("lstm_optimal_96points_prediction_4days.csv", DataFrame)[:, 5:2:end]

t0_ = [46, 116, 271, 351]
for k = eachindex(t0_)
    plot(
        heatmap(to2(test[90+t0_[k], :]'), color = RdYlBu_r,),
        heatmap(reshape(collect(minecraft[k, :]), 6, :), color = RdYlBu_r,),
        heatmap(to2(rcv(f(P_i_[t0_[k]])[90,:])'), color = RdYlBu_r,),
        layout = (:, 1), ticks = [], colorbar = false, margin = -1.6mm, size = [250, 150]
    )
    png("t0=$(t0_[k])")
end
for t0 in [46, 116, 271, 351]
    D[D .≥ Date(2025)][t0 .+ [0, 90]] |> println
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