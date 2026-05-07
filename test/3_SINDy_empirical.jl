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

t1 = 1
fttd = solve(f, [trng_z[t1, last(vrbl)]...], 0:0.01:365)[101:100:end, Not(1)]
plt_z1_1 = plot(trng_z.z1[t1:end], xlims = [0, 365]); plot!(fttd[:, 1], title = "fitted in 2019")
rsdl_1 = fttd[:] - trng_z.z1[t1 .+ (1:365)]
plt_rsdl_1 = plot(rsdl_1, color = :black, xlims = [0, 365])

t1 = 1 + 365
fttd = solve(f, [trng_z[t1, last(vrbl)]...], 0:0.01:365)[101:100:end, Not(1)]
plt_z1_2 = plot(trng_z.z1[t1:end], xlims = [0, 365]); plot!(fttd[:, 1], title = "fitted in 2020")
rsdl_2 = fttd[:] - trng_z.z1[t1 .+ (1:365)]
plt_rsdl_2 = plot(rsdl_2, color = :black, xlims = [0, 365])

t1 = 1 + 2*365
fttd = solve(f, [trng_z[t1, last(vrbl)]...], 0:0.01:365)[101:100:end, Not(1)]
plt_z1_3 = plot(trng_z.z1[t1:end], xlims = [0, 365]); plot!(fttd[:, 1], title = "fitted in 2021")
rsdl_3 = fttd[:] - trng_z.z1[t1 .+ (1:365)]
plt_rsdl_3 = plot(rsdl_3, color = :black, xlims = [0, 365])

t1 = 1 + 3*365
fttd = solve(f, [trng_z[t1, last(vrbl)]...], 0:0.01:365)[101:100:end, Not(1)]
plt_z1_4 = plot(trng_z.z1[t1:end], xlims = [0, 365]); plot!(fttd[:, 1], title = "fitted in 2022")
rsdl_4 = fttd[:] - trng_z.z1[t1 .+ (1:365)]
plt_rsdl_4 = plot(rsdl_4, color = :black, xlims = [0, 365])

avg_rsdl = (rsdl_1 + rsdl_2 + rsdl_3 + rsdl_4)/4

plot(plt_z1_1, plt_z1_2, plt_z1_3, plt_z1_4, plt_rsdl_1, plt_rsdl_2, plt_rsdl_3, plt_rsdl_4,
layout = (2, 4), legend = :none, size = [1200, 600])
plot(avg_rsdl, title = "average residual", color = :black, legend = :none)


prdt = solve(f, [test_z[1, last(vrbl)]...], 0:0.01:365)[101:100:end, Not(1)]
plt_z1_1 = plot(test_z.z1[1:end], xlims = [0, 365], label = "test data")
plot!(prdt[:], label = "SINDy only")
plot!(prdt[:] + avg_rsdl, label = "SINDy + average residual")

errors = []
for t1 = 1:335
    prdt = solve(f, [test_z[t1, last(vrbl)]...], 0:0.01:30)[101:100:end, Not(1)]
    prdt .= prdt[:] + avg_rsdl[t1 .+ (1:30)]
    plt_z1 = plot(test_z.z1[t1:end], xlims = [0, 30]); plot!(prdt[:, 1])
    push!(errors, rmse(prdt, test_z.z1[t1 .+ (1:30)], dims = 2)[:])
end
mean(stack(errors), dims = 2)[:][[1, 7, 30]]

