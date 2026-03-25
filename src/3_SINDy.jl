include("0_utils.jl")
include("1_datacall.jl")
include("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core/header.jl")

"""'''''''''''''''''''''''''''''''''''''''''''''

                SINDy for z1

'''''''''''''''''''''''''''''''''''''''''''''"""
dtemp = add_diff(rename(add_diff(data[:, [1]], method = :TVD), ["v", "u"]), method = :TVD)[:, ["dv", "v"]]
ddata = add_diff([dtemp[:, ["v"]] data[:, [1]]], method = :TVD)

trng_z = ddata[T .< Date(2023, 1, 1), :]
test_z = ddata[T .≥ Date(2023, 1, 1), :]

vrbl = getindex.(variablenames(trng_z), Ref(1:2))

cnfg = cook(last(vrbl); poly = 0:2)
f = SINDy(trng_z, vrbl, cnfg); # f |> print

prdt = solve(f, [test_z[t1, last(vrbl)]...], 0:0.01:30)[101:100:end, Not(1)]
plt_z1_1 = plot(test_z.z1[t1:end], xlims = [0, 30]); plot!(prdt[:, 1])
plt_z1_2 = plot(prdt - test_z.z1[t1 .+ (1:30)], xlims = [0, 30])


errors = []
for t1 = 1:10:365
    prdt = solve(f, [test_z[t1, last(vrbl)]...], 0:0.01:30)[101:100:end, Not(1)]
    plt_z1 = plot(test_z.z1[t1:end], xlims = [0, 30]); plot!(prdt[:, 1])
    push!(errors, rmse(prdt, test_z.z1[t1 .+ (1:30)], dims = 2)[:])
end
mean(stack(errors), dims = 2)[:]


"""'''''''''''''''''''''''''''''''''''''''''''''

                SINDy for all z

'''''''''''''''''''''''''''''''''''''''''''''"""
resultSINDy = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [])
for k in 1:150 push!(resultSINDy, ["", 0, 0, 0, 0, 0, 0]) end
f_ = [SINDy() for _ in 1:150]
prdt__ = [[] for _ in 1:150]
@showprogress @threads for k = 1:150
    dtemp = add_diff(rename(add_diff(data[:, [k]], method = :TVD), ["v", "u"]), method = :TVD)[:, ["dv", "v"]]
    ddata = add_diff([dtemp[:, ["v"]] data[:, [k]]], method = :TVD)

    trng_z = ddata[T .< Date(2023, 1, 1), :]
    test_z = ddata[T .≥ Date(2023, 1, 1), :]

    vrbl = getindex.(variablenames(trng_z), Ref(1:2))

    cnfg = cook(last(vrbl); poly = 0:1)
    f = SINDy(trng_z, vrbl, cnfg); # f |> print
    f_[k] = f

    resultSINDy365 = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [])
    for k in 1:365 push!(resultSINDy365, ["", 0, 0, 0, 0, 0, 0]) end
    errors = [[] for _ in 1:365]
    prdt_ = zeros(365)
    for t1 = 1:365
        prdt = solve(f, [test_z[t1, last(vrbl)]...], 0:0.01:30)[101:100:end, Not(1)]
        prdt_[t1] = prdt[1]
        errors[t1] = rmse(prdt, test_z[t1 .+ (1:30), end], dims = 2)[:]
        error1, error7, error30 = round.([errors[t1][1], errors[t1][7], errors[t1][30]], digits = 3)
        resultSINDy365[t1, :] = ["SINDy with z$k", t1, 1, f.r2, error1, error7, error30]
    end
    prdt__[k] = prdt_
    CSV.write("G:/seasurface/260325/resultSINDy_z$k.csv", resultSINDy365)
    rmse_ = mean(stack(errors), dims = 2)[:]
    rmse1, rmse7, rmse30 = round.([rmse_[1], rmse_[7], rmse_[30]], digits = 3)
    resultSINDy[k, :] = ["SINDy with z$k", -1, 1, f.r2, rmse1, rmse7, rmse30]
    CSV.write("G:/seasurface/260325/resultSINDy.csv", resultSINDy)
end

cprA = dataframe2tensor(data[Date(2023, 1, 2) .≤ T .≤ Date(2024, 1, 1), :])
cprB = reshape(stack(prdt__)', 10, 15, :)
# rmse(cprA[1,1,:], cprB[1,1,:], dims = 2)
# rmse(cprA, cprB)
anime = @animate for t1 = 1:365
    rmse1 = round(rmse(cprA[:, :, t1], cprB[:, :, t1]), digits = 3)
    plot(
        heatmap(cprA[:, :, t1]),
        heatmap(cprB[:, :, t1]),
        heatmap(cprA[:, :, t1] - cprB[:, :, t1], color = :balance, clims = (-1, 1)),
        layout = (1, 3), size = (1200, 400),
        plot_title = "$(T[Date(2023, 1, 2) .≤ T .≤ Date(2024, 1, 1)][t1]), rmse1 = $rmse1",
    )
end
mp4(anime, "G:/seasurface/260325/movie.mp4", fps = 4)


"""'''''''''''''''''''''''''''''''''''''''''''''

            Convolutional SINDy

'''''''''''''''''''''''''''''''''''''''''''''"""

function splash(k)
    bit_matrix = zeros(Bool, 10, 15)
    bit_matrix[k] = true
    for ij in Ref([findfirst(bit_matrix).I...]) .+ [[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]]
        try
            bit_matrix[ij...] = true
        catch
        end
    end
    return findall(bit_matrix[:])
end
splash(75)
whereis(splash(75))

resultSINDy = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [])
for k in 1:150 push!(resultSINDy, ["", 0, 0, 0, 0, 0, 0]); push!(f_, ) end
f_ = [SINDy() for _ in 1:150]
prdt__ = [[] for _ in 1:150]
@showprogress @threads for k = 1:150
    k_ = splash(k)
    dim = length(k_)
    
    dtemp = add_diff(data[:, k_], method = :TVD)
    ddtemp = add_diff(dtemp[:, 1:dim], method = :TVD)
    ddata = [ddtemp data[:, k_]]

    trng_z = ddata[T .< Date(2023, 1, 1), :]
    test_z = ddata[T .≥ Date(2023, 1, 1), :]

    # vrbl = getindex.(variablenames(trng_z), Ref(1:2))
    # vrbl = (names(ddtemp), names(data)[k_])
    vrbl = (names(ddtemp), names(dtemp))

    cnfg = cook(last(vrbl); poly = 0:2)
    f = SINDy(trng_z, vrbl, cnfg); # f |> print
    f_[k] = f

    resultSINDy365 = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [])
    for k in 1:365 push!(resultSINDy365, ["", 0, 0, 0, 0, 0, 0]) end
    errors = [[] for _ in 1:365]
    prdt_ = zeros(365)
    for t1 = 1:365
        prdt = DataFrame(solve(f, [test_z[t1, last(vrbl)]...], 0:0.01:30)[101:100:end, (dim+1):end], names(data[:, k_]))[:, names(data)[k]]
        prdt_[t1] = prdt[1]
        errors[t1] = rmse(prdt, test_z[t1 .+ (1:30), end], dims = 2)[:]
        error1, error7, error30 = round.([errors[t1][1], errors[t1][7], errors[t1][30]], digits = 3)
        resultSINDy365[t1, :] = ["SINDy with z$k", t1, 1, f.r2, error1, error7, error30]
    end
    prdt__[k] = prdt_
    CSV.write("G:/seasurface/260324/resultSINDy_z$k.csv", resultSINDy365)
    rmse_ = mean(stack(errors), dims = 2)[:]
    rmse1, rmse7, rmse30 = round.([rmse_[1], rmse_[7], rmse_[30]], digits = 3)
    resultSINDy[k, :] = ["SINDy with z$k", -1, 1, f.r2, rmse1, rmse7, rmse30]
    CSV.write("G:/seasurface/260324/resultSINDy.csv", resultSINDy)
end
