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

plt_ngrc = plot(xticks = [0, 30, 60, 90, 120], yticks = [0.3, 0.5, 0.7], ylims = [0.3, 0.7],
    legend = (0.1, 0.9), background_color_legend=nothing, foreground_color_legend=nothing)
bilinear = Matrix(CSV.read("lstm_mae_by_date_lead_bilinear157126.csv", DataFrame)[:, 3:end])
μ = mean.(eachcol(bilinear))
σ = std.(eachcol(bilinear))/5
plot!(plt_ngrc, μ, color = :dodgerblue, lw = 2, label = "interpolation")
plot!(plt_ngrc, μ .- σ, fillrange = μ .+ σ, fillcolor = :dodgerblue, fillalpha = 0.2, label = :none, alpha = 0)
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
    layout = (:, 1), size = [300, 400]
)


"""''''''''''''''''''''''''''''''''''''''''''''''''''

                bilinear interpolation

''''''''''''''''''''''''''''''''''''''''''''''''''"""

using Interpolations

x = [1.0, 2.0, 3.5, 5.0]   # 불균일 간격 가능
y = [1.0, 2.5, 4.0]
z = rand(length(x), length(y))

Z = reshape(collect(minecraft[2, :]), 6, 16)

itp = interpolate((
    round.(Int64, range(1, 251, length = 6)),
    round.(Int64, range(1, 626, length = 16)),
    ), Z, Gridded(Linear()))


value = [itp(i, j) for i in 1:251, j in 1:626]

heatmap(value)


minecraft = CSV.read("lstm_optimal_96points_prediction_4days.csv", DataFrame)[:, 5:2:end]
t0_ = [46, 116, 271, 351]
for k = eachindex(t0_)
    Z = reshape(collect(minecraft[k, :]), 6, 16)
    itp = interpolate((
        round.(Int64, range(1, 251, length = 6)),
        round.(Int64, range(1, 626, length = 16)),
        ), Z, Gridded(Linear()))
    Z1 = [itp(i, j) for i in 1:251, j in 1:626]
    Z0 = to2(test[90+t0_[k], :]')
    Z2 = to2(rcv(f(P_i_[t0_[k]])[90,:])')
    plot(
        heatmap(Z0, color = RdYlBu_r,),
        # heatmap(reshape(collect(minecraft[k, :]), 6, :), color = RdYlBu_r,),
        heatmap(Z1, color = RdYlBu_r,),
        heatmap(Z2, color = RdYlBu_r,),
        layout = (:, 1), ticks = [], colorbar = false, margin = -1.6mm, size = [250, 150]
    )
    @info "t0=$(t0_[k])" mean(abs, skipmissing(Z0 - Z1)), mean(abs, skipmissing(Z0 - Z2))
    png("t0=$(t0_[k])")
end
for t0 in [46, 116, 271, 351]
    D[D .≥ Date(2025)][t0 .+ [0, 90]] |> println
end

