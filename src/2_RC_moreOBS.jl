include("0_utils.jl")
include("1_datacall.jl")

id_observer = [1, 10, 141, 150]
id_observer = setdiff(1:150, id_observer)
whereis(ids) = reshape([k ∈ ids for k in 1:150], 10, 15)
whereis(id_observer)

resultRC_ = [DataFrame() for _ in 1:365]
t0 = 1

[T data][tspan, :]
# @showprogress @threads for t0 = (1:365) # = rand(1:365)
    tspan = (t0:findfirst(x -> x == (T[t0] + Year(3) - Day(1)), T))

    _U = Matrix(data[tspan, id_observer])'
    _S = Matrix(data[tspan, Not(id_observer)])'
    newU = Matrix(data[last(tspan) .+ (0:30), id_observer])'
    actS = Matrix(data[last(tspan) .+ (0:30), Not(id_observer)])'

    resultRC = DataFrame(Model = [], t0 = [], input = [], rmse1 = [], rmse7 = [], rmse30 = [], Hyperparameter = [])
    for _ in 1:10
        push!(resultRC, ["", 0, 0, 0, 0, 0, 0, []])
    end
    for k in 1:nrow(resultRC)
        N = 1000; α = 1; ρ = 1
        β = (120 < t0 ≤ 270) ? 1e-6 : 1e+1

        _rc = reservoir_computing(_U, _S; N, α, β, ρ)
        newS = _rc(newU)
        _rc = reservoir_computing(_S, _S; N, α, β, ρ)
        newS = _rc(newS)
        rmse_ = vec(rmse(newS, actS, dims = 1)); plot(rmse_)
        rmse1, rmse7, rmse30 = round.([rmse_[1], rmse_[7], rmse_[30]], digits = 3)

        resultRC[k, :] = ["RC with seasonal beta", t0, length(tspan), rmse1, rmse7, rmse30, β]
    end
    resultRC_[t0] = resultRC
# end

sqrt(mean(abs2, actS[:, 1] - newS[:, 1]))
_rc |> propertynames
_rc.Win
_rc.Wout
_rc.r

data = CSV.read("lorenz.csv", DataFrame)[1:10:end, :]
standardize(x) = (x .- mean(x)) ./ std(x)
for col in eachcol(data)
    col .= standardize(col)
end
_U = [data.x data.y]'
_S = data.z'
@time rc_ = reservoir_computing(_U[:, 1:9000], _S[:, 1:9000], N = 1000, D = 10, α = 1)
plot(vec(rc_(_U[:, 9001:end])))
plot!(vec(_S[:, 9001:end]))