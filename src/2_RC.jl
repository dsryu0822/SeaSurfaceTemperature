include("0_utils.jl")
include("1_datacall.jl")


# t0 = 1
id_observer = [1, 10, 141, 150]
# tspan = t0:findlast(T .< Date(2023, 01, 08))
# _U = Matrix(data[tspan, id_observer])';
# _S = Matrix(data[tspan, Not(id_observer)])';
# newU = Matrix(data[last(tspan) .+ (1:30), id_observer])';
# actS = Matrix(data[last(tspan) .+ (1:30), Not(id_observer)])';

# rc = reservoir_computing(_U, _S; α = 1)
# newS = rc(newU);
# # newX = [newS; newU][sortperm([setdiff(1:150, id_observer); id_observer]), :];
# rmse_ = vec(rmse(newS, actS, dims = 1));
# rmse1, rmse7, rmse30 = round.([rmse_[1], rmse_[7], rmse_[30]], digits = 2)

# plot(actS[4, :])
# plot!(newS[4, :])

# heatmap(actS[1:8, :])
# heatmap(newS[1:8, :])
# heatmap((actS - newS)[1:8, :])



A = Matrix(LSTMpred[:, 2:end])
B = Matrix(LSTMpred[:, 2:end])
sqrt(mean(abs2, A - B))

# LSTMpred = CSV.read("lstm_prediction_4points.csv", DataFrame)[:, 1:2:end]
LSTMpred = [T data][1469:end, [1; id_observer .+ 1]]
# [T data][findfirst(Date(2020, 1, 7) .== T):end, :]
resultRC_ = [DataFrame() for _ in 1:365]
@showprogress @threads for t0 = (1:365) # = rand(1:365)
    # tspan = t0:findfirst(x -> x == (T[t0] + Year(3)), T)
    tspan = (t0:findfirst(x -> x == (T[t0] + Year(3)), T)) .+ 371

    _U = Matrix(data[tspan, id_observer])'
    _S = Matrix(data[tspan, Not(id_observer)])'
    # newU = Matrix(data[last(tspan) .+ (1:30), id_observer])'
    newU = Matrix(LSTMpred[t0 .+ (1:30) .- 1, 2:end])'
    actS = Matrix(data[last(tspan) .+ (1:30), Not(id_observer)])'

    resultRC = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [], Hyperparameter = [])
    for _ in 1:10
        push!(resultRC, ["", 0, 0, 0, 0, 0, 0, []])
    end
    for k in 1:nrow(resultRC)
        N = 1000; # rand(10:1000)
        α = 1; # exp10(-rand())
        # β = exp10(3 - 9rand())
        β = (120 < t0 ≤ 270) ? 1e-6 : 1e+1
        ρ = 1; # 2rand()

        _rc = reservoir_computing(_U, _S; N, α, β, ρ)
        newS = _rc(newU)
        rmse_ = vec(rmse(newS, actS, dims = 1)); # plot(rmse_)
        rmse1, rmse7, rmse30 = round.([rmse_[1], rmse_[7], rmse_[30]], digits = 3)

        resultRC[k, :] = ["RC with seasonal beta", t0, 365*3, 0.0, rmse1, rmse7, rmse30, β]
    end
    resultRC_[t0] = resultRC
    CSV.write("G:/seasurface/260320/resultRC.csv", [resultRC_...;], bom = true)
end
# istaskdone(task_1)
# count(!isempty, resultRC.Model)
# sort(resultRC, :t0)


# for resultRC in resultRC_
#     # resultRC.N = getproperty.(resultRC.Hyperparameter, :N)
#     # resultRC.α = getproperty.(resultRC.Hyperparameter, :α)
#     resultRC.β = getproperty.(resultRC.Hyperparameter, :β)
#     # resultRC.ρ = getproperty.(resultRC.Hyperparameter, :ρ)
#     # plot(scatter(resultRC.N, resultRC.rmse30, xlabel = "N")
#     # , scatter(resultRC.α, resultRC.rmse30, xscale = :log10, xlabel = "α")
#     # , scatter(resultRC.β, resultRC.rmse30, xscale = :log10, xlabel = "β")
#     # , scatter(resultRC.ρ, resultRC.rmse30, xlabel = "ρ")
#     # , legend = :none, yscale = :log10, msw = 0, ylims = [2e-1, 2e+0]
#     # )
#     t0 = first(resultRC.t0)
#     scatter(resultRC.β, resultRC.rmse30, xscale = :log10, xlabel = "β($t0)")
#     png("G:/seasurface/260318/$(resultRC.t0[1])")
# end

# scatter(mod.(resultRC.t0, 365), resultRC.rmse30)
# plot(sum(abs2, diff(actS, dims = 2), dims = 2))
# plot(plot(rmse_), plot(vec(std(actS, dims = 1))), layout = (2, 1))

# CSV.write("G:/seasurface/260318/resultRC.csv", [resultRC_...;], bom = true)


# """'''''''''''''''''''''''''''''''''''''''''''''

#                 LSTM observation        

# '''''''''''''''''''''''''''''''''''''''''''''"""
# LSTMpred = CSV.read("lstm_prediction_4points.csv", DataFrame)[:, 1:2:end]
# newU2 = Matrix(LSTMpred[1:30, Not(:date)])'
# newS2 = g(newU2)
# _rmse2 = vec(rmse(newS2', actS'))

# newX2 = [newS2; newU2][sortperm([setdiff(1:150, id_observer); id_observer]), :]

# rmse1, rmse7, rmse30 = round.([_rmse2[1], _rmse2[7], _rmse2[30]], digits = 2)
# Y0 = tnsr[:, :, last(tspan) + t]
# Y1 = reshape(newX2, 10, 15, :)[:, :, t]
# plt_rmse2 = plot(_rmse2, color = :black, legend = :none)
# scatter!(plt_rmse2, [t], [_rmse2[t]])
# plot(
#     heatmap(Y0, clims = extrema([Y0; Y1])),
#     heatmap(Y1, clims = extrema([Y0; Y1])),
#     heatmap(Y0 - Y1, clims = (-5, 5), color = [:blue, :white, :red]),
#     plt_rmse2,
#     plot_title = "err1: $(rmse1), err7: $(rmse7), err30: $(rmse30)",
# )


# """'''''''''''''''''''''''''''''''''''''''''''''

#             Parameter tuning of RC

# '''''''''''''''''''''''''''''''''''''''''''''"""

# resultRC = CSV.read("G:/seasurface/260318/resultRC.csv", DataFrame)
# mean(resultRC.rmse30)

# t0__ = []
# rmse30__ = []
# beta__ = []
# for df = groupby(resultRC, :t0)
#     _df = sort(df, :rmse30)
#     push!(t0__, _df.t0[1:10])
#     push!(rmse30__, _df.rmse30[1:10])
#     push!(beta__, _df.Hyperparameter[1:10])
#     # scatter(df.Hyperparameter, df.rmse30, xscale = :log10, ylims = [0, 1.5], xticks = [1e-6, 1e-3, 1e+0, 1e+3], yticks = [0, .1, .2, .3, .4, .5, 1., 1.5], legend = :none, msw = 0, color = :black, title = L"t_0 = %$(df.t0[1])", xlabel = L"\beta", ylabel = "MSE(30)")
#     # png("G:/seasurface/260318/details/$(df.t0[1]).png")
# end
# using LaTeXStrings
# scatter([t0__...;], [beta__...;], yscale = :log10, xticks = [0, 90, 180, 270, 365], xlabel = "t0", ylabel = L"\beta")
# scatter([t0__...;], [rmse30__...;], yscale = :log10, xticks = [0, 90, 180, 270, 365], xlabel = "t0", ylabel = "RMSE(30)")


# scatter(median.(beta__), yscale = :log10)
# mean([rmse30__...;])
# mean([rmse30__[1:90]...;])
# mean([rmse30__[91:180]...;])
# mean([rmse30__[181:270]...;])
# mean([rmse30__[271:365]...;])
# mean([rmse30__[Not(181:270)]...;])

# plot(tnsr[1, 1, 150:190])
# plot(tnsr[10, 1, 150:190])
# plot(tnsr[1, 15, 150:190])
# plot(tnsr[10, 15, 150:190])
# plot(
#     heatmap(tnsr[:, :, 157+29]),
#     heatmap(tnsr[:, :, 157+30]),
#     heatmap(tnsr[:, :, 157+31]),
#     heatmap(tnsr[:, :, 157+32]),
#     layout = (4, 1), size = [600, 1400]
# )

# argsim(x0, x) = argmin(abs.(x .- x0))

# bar_ = []
# for beta = logrange(1e-6, 1e+3, 20)
#     println(beta)
#     foo = argsim.(beta, getproperty.([groupby(resultRC, :t0)...], :Hyperparameter))
#     bar = mean([groupby(resultRC, :t0)[i].rmse30[foo[i]] for i in 1:365])
#     push!(bar_, bar)
# end
# scatter(logrange(1e-6, 1e+3, 20), bar_, xscale = :log10, xlabel = L"\beta", ylabel = "mean RMSE(30)", title = "Best β across t0", legend = :none, yticks = 0:0.1:1)


# histogram(log10.([beta__[1:90]...;]), bin = 20)
# histogram(log10.([beta__[90:180]...;]), bin = 20)
# histogram(log10.([beta__[180:270]...;]), bin = 20)
# histogram(log10.([beta__[270:365]...;]), bin = 20)

# plot([histogram(log10.([beta__[t0:(t0+29)]...;]), bin = 20, color = :black) for t0 in 1:30:350]..., size = [1200, 800], xticks = [-6, -3, 0, 1, 3], ylims = [0, 70], layout = (4, 3), legend = :none)

# default()
heatmap(tnsr[:, :, 1])
heatmap(tnsr[:, :, 1][:])