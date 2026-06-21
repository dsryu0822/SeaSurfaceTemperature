include("0_utils.jl")
include("1_datacall.jl")
include.(readdir("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core", join=true))

to3(array2) = reshape(Matrix(array2)', length(X), length(Y), :)
findindex(big, small) = [findfirst(==(x), big) for x in small]

candy_observer = sort([1,31301,62601,93901,125201,156501,42,31342,62642,93942,125242,156542,84,31384,62684,93984,125284,156584,126,31426,62726,94026,125326,156626,167,31467,62767,94067,125367,156667,209,31509,62809,94109,125409,156709,251,31551,62851,94151,125451,156751,292,31592,62892,94192,125492,156792,334,31634,62934,94234,125534,156834,376,31676,62976,94276,125576,156876,417,31717,63017,94317,125617,156917,459,31759,63059,94359,125659,156959,501,31801,63101,94401,125701,157001,542,31842,63142,94442,125742,157042,584,31884,63184,94484,125784,157084,626,31926,63226,94526,125826,157126])
trng = Matrix(nino34[(T .< Date(2023)), :])
vldn = Matrix(nino34[(Date(2023) .≤ T .< Date(2024)), :])

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
result = CSV.read("NGRC_sweep_result.csv", DataFrame)
result[argmin(result.mae), :]; sort(result.id_observer[argmin(result.mae)])

lstm = CSV.read("lstm_optimal_points_prediction_1.csv", DataFrame)
lstm = lstm[lstm.date .< Date(2026), :]
_trng = Matrix(nino34[(T .< Date(2024)), :])
test = Matrix(nino34[(Date(2025) .≤ T), :])

id_optimal = [209, 31342, 31509, 31634, 63142, 93901, 94026, 94359, 125367, 125659, 125701, 156709]
id_target = setdiff(axes(nino34, 2), [id_optimal; id_missing])
vrbl = (id_target, id_optimal)
_id_optimal = findindex(candy_observer, id_optimal)

ngrc_ = ngrc_sweep(_trng, vrbl...;)
f = ngrc(_trng, vrbl...; λ = 100)

P_i_ = [Matrix(df[:, 5:2:end]) for df in groupby(lstm, :date)]
lead_ = ["lead$(lpad(d, 3, "0"))" for d in 1:120]
pfmc = DataFrame([[] for _ in lead_], lead_)
@showprogress for k in 1:365
    actl = test[k .+ (1:120), id_target]
    pred = f(P_i_[k])
    push!(pfmc, mean(abs, actl - pred , dims = 2))
end
plot(mean.(eachcol(pfmc)), xlabel = "lead", ylabel = "average MAE")
plot(mean(abs, lstm[:, 4:2:end] - lstm[:, 5:2:end], dims = 2)[:], color = :black)

mae(f(P_i_[k]), test[k .+ (1:120), id_target])

test
mae(lstm[:, 5:2:end][:, _id_optimal], lstm[:, 4:2:end][:, _id_optimal])

plot(mean(abs2, f(P_i_[30]) - test[30 .+ (1:120), id_target], dims = 2))

allpoints = collect(Base.product(1:626, 1:251))
candypoints = allpoints[candy_observer]
optimalpoints = allpoints[id_optimal]
heatmap(to3(test)[:,:,1])
scatter!(first.(optimalpoints), last.(optimalpoints), color = :blue, msc = :white)

asdf = f(test[1:365, id_optimal])
mae(asdf, dw(test[1:365, id_target]))
qwer = f(lstm[lstm.lead .== 0, 4:2:end][:, _id_optimal])
mae(qwer, dw(test[1:365, id_target]))

lstm[lstm.lead .== 0,[1; 4:2:end]]
lstm[lstm.lead .== 0,[1; 4:2:end]][:, 1 .+ [0; _id_optimal]]
test[:, candy_observer]

test[:, id_optimal]
aaa = nino34[(Date(2025) .≤ T), candy_observer]
bbb = lstm[lstm.lead .== 0, 4:2:end]
for k in eachindex(candy_observer)
    plot(xlims = [1, 365], title = "observer index: $(candy_observer[k])")
    plot!(aaa[:,k])
    plot!(bbb[:,k])
    png("index = $(k)_observer index $(candy_observer[k]).png")
end

_lstm = Matrix(lstm[lstm.lead .== 0, 4:2:end])
thth = DataFrame(this = [], that = [])
@showprogress for k in 1:96
    qwer = mean(abs, test .- _lstm[:, k], dims = 1)
    qwer[ismissing.(qwer)] .= Inf
    this = argmin(qwer)[2]
    that = candy_observer[k]
    push!(thth, (this, that))
end
thth.name = names(lstm[lstm.lead .== 0, 4:2:end])
CSV.write("index.csv", thth)
test[:, this]
names(lstm[lstm.lead .== 0, 4:2:end])

[test[:, 10292] _lstm[:, 2]]
[test[:, 20834] _lstm[:, 3]]
[test[:, 31376] _lstm[:, 4]]
[test[:, 41667] _lstm[:, 5]]

allpoints[[10292, 20834, 31376]]

thth.this
wrongpoints = allpoints[thth.this]

heatmap(to3(test)[:,:,1])
scatter!(first.(candypoints), last.(candypoints), legend = :none, color = :white)

heatmap(to3(test)[:,:,1])
scatter!(first.(wrongpoints), last.(wrongpoints), legend = :none, color = :white)

CSV.write("optimal_points.csv", [DataFrame(T = T) nino34[:, id_optimal]])
