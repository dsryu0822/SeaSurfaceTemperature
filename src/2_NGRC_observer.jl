ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,5]])
include("0_utils.jl")
include("1_datacall.jl")

to3(array2) = reshape(Matrix(array2)', length(X), length(Y), :)
to2(array1) = reshape(array1, length(X), length(Y))
findindex(big, small) = [findfirst(==(x), big) for x in small]

candy_observer = sort([1,31301,62601,93901,125201,156501,42,31342,62642,93942,125242,156542,84,31384,62684,93984,125284,156584,126,31426,62726,94026,125326,156626,167,31467,62767,94067,125367,156667,209,31509,62809,94109,125409,156709,251,31551,62851,94151,125451,156751,292,31592,62892,94192,125492,156792,334,31634,62934,94234,125534,156834,376,31676,62976,94276,125576,156876,417,31717,63017,94317,125617,156917,459,31759,63059,94359,125659,156959,501,31801,63101,94401,125701,157001,542,31842,63142,94442,125742,157042,584,31884,63184,94484,125784,157084,626,31926,63226,94526,125826,157126])
trng = Matrix(nino34[(T .< Date(2023)), :])
vldn = Matrix(nino34[(Date(2023) .≤ T .< Date(2024)), :])

begin
    λ_ = exp10.(-6:3)
    result = DataFrame(n_obs = Int64[], bestλ = Float64[], mae = Float64[], rmse = Float64[], id_observer = [])
    _id_observer = Int64[]
    if isfile("NGRC_sweep_result.csv")
        result = CSV.read("NGRC_sweep_result.csv", DataFrame)
        result.id_observer .= [eval(Meta.parse(id_)) for id_ in result.id_observer]
        _id_observer = result.id_observer[argmin(result.mae)]
    end
    for n_obs = 1:15
        @info "$n_obs phase started"
        @showprogress for candy = candy_observer
            id_observer = [_id_observer; candy]
            if (length(unique(id_observer)) .!= length(id_observer)) continue end
            id_target = setdiff(axes(nino34, 2), [id_observer; id_missing])
            vrbl = (id_target, id_observer)
            P_i = vldn[:, last(vrbl)]
            P_o = dw(vldn[:, first(vrbl)])

            f_ = ngrc_sweep(trng, vrbl...;)
            mae_ = zeros(length(λ_))
            rmse_ = zeros(length(λ_))
            @threads for k in eachindex(λ_)
                λ = λ_[k]
                @time f = NGRC(f_(λ), λ)
                @time fP_i = f(P_i)
                @time mae_[k] = mae(fP_i, P_o)
                @time rmse_[k] = rmse(fP_i, P_o)
            end
            push!(result, (n_obs, λ_[argmin(mae_)], minimum(mae_), rmse_[argmin(mae_)], id_observer))
            CSV.write("NGRC_sweep_result.csv", result)
        end
        presult = result[result.n_obs .== n_obs, :]
        global _id_observer = presult.id_observer[argmin(presult.mae)]
    end
end
result = CSV.read("NGRC_sweep_result.csv", DataFrame)
result.id_observer .= [eval(Meta.parse(id_)) for id_ in result.id_observer]
result[argmin(result.mae), :]

bestresult = vcat(DataFrame.(first.([sort(df, :mae) for df in groupby(result, :n_obs)]))...)
abs.([0; diff(bestresult.mae)]) ./ bestresult.mae
