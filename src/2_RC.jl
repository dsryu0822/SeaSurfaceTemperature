include("0_utils.jl")
include("1_datacall.jl")


t0 = 1
id_observer = [1, 10, 141, 150]
tspan = t0:findlast(T .< Date(2023, 01, 08))
_U = Matrix(data[tspan, id_observer])';
_S = Matrix(data[tspan, Not(id_observer)])';
newU = Matrix(data[last(tspan) .+ (1:30), id_observer])';
actS = Matrix(data[last(tspan) .+ (1:30), Not(id_observer)])';

rc = reservoir_computing(_U, _S; α = 1)
newS = rc(newU);
# newX = [newS; newU][sortperm([setdiff(1:150, id_observer); id_observer]), :];
rmse_ = vec(rmse(newS, actS, dims = 1));
rmse1, rmse7, rmse30 = round.([rmse_[1], rmse_[7], rmse_[30]], digits = 2)

plot(actS[4, :])
plot!(newS[4, :])

heatmap(actS[1:8, :])
heatmap(newS[1:8, :])
heatmap((actS - newS)[1:8, :])

resultRC_ = []
@showprogress @threads for t0 = 1:365
    # t0 = rand(1:365)
    tspan = t0:findfirst(x -> x == (T[t0] + Year(3)), T)

    _U = Matrix(data[tspan, id_observer])'
    _S = Matrix(data[tspan, Not(id_observer)])'
    newU = Matrix(data[last(tspan) .+ (1:30), id_observer])'
    actS = Matrix(data[last(tspan) .+ (1:30), Not(id_observer)])'

    resultRC = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [], Hyperparameter = [])
    for _ in 1:100
        push!(resultRC, ["", 0, 0, 0, 0, 0, 0, []])
    end
    task_1 = for k in 1:nrow(resultRC)
        N = 1000; # rand(10:1000)
        α = 1; # exp10(-rand())
        β = exp10(2 - 8rand())
        ρ = 2rand()

        _rc = reservoir_computing(_U, _S; N, α, β, ρ)
        newS = _rc(newU)
        rmse_ = vec(rmse(newS, actS, dims = 1)); plot(rmse_)
        rmse1, rmse7, rmse30 = round.([rmse_[1], rmse_[7], rmse_[30]], digits = 2)

        resultRC[k, :] = ["RC", t0, 365*4, 0.0, rmse1, rmse7, rmse30, (N = N, α = α, β = β, ρ = ρ)]
    end
    push!(resultRC_, resultRC)
end
istaskdone(task_1)
count(!isempty, resultRC.Model)
sort(resultRC, :t0)


for resultRC in resultRC_
    resultRC.N = getproperty.(resultRC.Hyperparameter, :N)
    resultRC.α = getproperty.(resultRC.Hyperparameter, :α)
    resultRC.β = getproperty.(resultRC.Hyperparameter, :β)
    resultRC.ρ = getproperty.(resultRC.Hyperparameter, :ρ)
    sort(resultRC[:, Not(:Hyperparameter)], :rmse1)
    plot(scatter(resultRC.N, resultRC.rmse30, xlabel = "N")
    , scatter(resultRC.α, resultRC.rmse30, xscale = :log10, xlabel = "α")
    , scatter(resultRC.β, resultRC.rmse30, xscale = :log10, xlabel = "β")
    , scatter(resultRC.ρ, resultRC.rmse30, xlabel = "ρ")
    , legend = :none, yscale = :log10, msw = 0, ylims = [2e-1, 2e+0]
    )
    png("G:/seasurface/260305/resultRC2kindparam/$(resultRC.t0[1])")
end

scatter(mod.(resultRC.t0, 365), resultRC.rmse30)
plot(sum(abs2, diff(actS, dims = 2), dims = 2))
plot(plot(rmse_), plot(vec(std(actS, dims = 1))), layout = (2, 1))

CSV.write("G:/seasurface/260305/resultRC2kindparam.csv", [resultRC_...;], bom = true)

"""'''''''''''''''''''''''''''''''''''''''''''''

                LSTM observation        

'''''''''''''''''''''''''''''''''''''''''''''"""

LSTMpred = CSV.read("lstm_prediction_4points.csv", DataFrame)[:, 1:2:end]
newU2 = Matrix(LSTMpred[1:30, Not(:date)])'
newS2 = g(newU2)
_rmse2 = vec(rmse(newS2', actS'))

newX2 = [newS2; newU2][sortperm([setdiff(1:150, id_observer); id_observer]), :]

rmse1, rmse7, rmse30 = round.([_rmse2[1], _rmse2[7], _rmse2[30]], digits = 2)
Y0 = tnsr[:, :, last(tspan) + t]
Y1 = reshape(newX2, 10, 15, :)[:, :, t]
plt_rmse2 = plot(_rmse2, color = :black, legend = :none)
scatter!(plt_rmse2, [t], [_rmse2[t]])
plot(
    heatmap(Y0, clims = extrema([Y0; Y1])),
    heatmap(Y1, clims = extrema([Y0; Y1])),
    heatmap(Y0 - Y1, clims = (-5, 5), color = [:blue, :white, :red]),
    plt_rmse2,
    plot_title = "err1: $(rmse1), err7: $(rmse7), err30: $(rmse30)",
)

