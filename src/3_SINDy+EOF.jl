include("0_utils.jl")
include("1_datacall.jl")
include("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core/header.jl")

# trng_z = data[T .< Date(2023, 1, 1), :]
test_z = data[T .≥ Date(2023, 1, 1), :]

eof = EOF(Matrix(data)); # rmse(Matrix(data), eof(eof.U))
[(eof.Σ / sum(eof.Σ)) (cumsum(eof.Σ) / sum(eof.Σ))]
temp_dv = add_diff(rename(add_diff(DataFrame(u = eof.U[:, 1]), method = :TVD), ["v", "u"]), method = :TVD)[:, ["dv", "v"]]
ddata = add_diff([temp_dv[:, ["v"]] DataFrame(eof.U, ["u$k" for k in 1:size(eof.U, 2)])], method = :TVD)
trng_u = ddata[T .< Date(2023, 1, 1), :]
test_u = ddata[T .≥ Date(2023, 1, 1), :]

candySINDy = DataFrame(nu = [], deg = [], SINDyloss = [])
@async @time for nu = 1:20
    # ddata = add_diff([temp_dv[:, ["v"]] DataFrame(eof.U, ["u$k" for k in 1:size(eof.U, 2)])][:, 1 .+ (0:nu)], method = :TVD)
    vrbl = getindex.(variablenames(ddata), Ref(1:(nu+1)))
    
    for deg = 1:1
        cnfg = cook(last(vrbl); poly = 0:deg)
        loss_ = zeros(nrow(test_z)-1)
        @threads for k in eachindex(loss_)
            f = SINDy(trng_u, vrbl, cnfg); # f |> print
            prdt = solve(f, [test_u[k, last(vrbl)]...], 0:0.01:1)[101:100:end, Not(1)]
            loss_[k] = rmse(eof(prdt), [test_z[k+1, :]...]')
        end
        push!(candySINDy, [nu, deg, mean(loss_)])
    end
end
candySINDy; plot(candySINDy.nu, candySINDy.SINDyloss, yscale = :log10)

nu = 5; vrbl = getindex.(variablenames(ddata), Ref(1:(nu+1)))
cnfg = cook(last(vrbl); poly = 0:1)
f = SINDy(trng_u, vrbl, cnfg); # f |> print

ic = [trng_u[end, last(vrbl)]...]
@time prdt = solve(f, ic, 0:0.01:365)[101:100:end, Not(1)]
ic_ = [[trng_u[end, last(vrbl)]...]]; [push!(ic_, [test_u[t, last(vrbl)]...]) for t in 1:364];
@time prdt = stack([[solve(f, ic, 0:0.01:1)[101:100:end, Not(1)]...] for ic in ic_])'
plt_u1 = plot(test_u.u1, xlims = [0, 30]); plot!(prdt[:, 1])
plt_u2 = plot(test_u.u2, xlims = [0, 30]); plot!(prdt[:, 2])
plt_u3 = plot(test_u.u3, xlims = [0, 30]); plot!(prdt[:, 3])
plot(plt_u1, plt_u2, plt_u3, layout = (3, 1), legend = :none, xlims = [0, 365])


qwer = tnsr[:, :, Date(2023, 1, 1) .≤ T .< Date(2024, 1, 1)]
asdf = dataframe2tensor(eof(prdt))
zxcv = qwer - asdf
plot(T[Date(2023, 1, 1) .≤ T .< Date(2024, 1, 1)], rmse(qwer, asdf, dims = [1,2])[:], ylabel = "RMSE1", color = :black, legend = :none)
t = 90; plot(
    heatmap(qwer[:,:,t]),
    heatmap(asdf[:,:,t]),
    heatmap(zxcv[:,:,t], color = :balance, clims = extrema(zxcv)),
    size = [1200, 400], layout = (1, 3), title = ["true" "predicted" "error"]
)

data.z1