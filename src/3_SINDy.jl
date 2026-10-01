ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,6]])
include("0_utils.jl")
include("1_datacall.jl")

id_ovserver = []
id_target = setdiff(axes(nino34, 2), [id_missing; id_ovserver])
rcv(x) = recover(x, (id_target, []))


# plot(T, mean(Matrix(nino34[:, id_target]), dims = 2), label = "mean", color = :black, xticks = :years)
trng = Matrix(nino34[(Date(2019) .≤ T .< Date(2023)), id_target])
μ = mean(trng)
trng = trng .- μ

@time eofed = eof(trng)
eof2Z(uu, nu) = uu * (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
eof2U(zz, nu) = zz / (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
plot(cumsum(eofed.Σ ./ sum(eofed.Σ)), xscale = :log10, label = :none, color = :black)

# U = [DataFrame(T = axes(trng, 1) ./ 365) DataFrame(eofed.U[:, 1:3], :auto)]
dfU = [
    add_diff(DataFrame(T = axes(trng, 1) ./ 365); method = :FDM) add_diff(DataFrame(eofed.U[:, 1:3], ["u$j" for j in 1:3]); method = :TVD)[1:(end-1), :]
]
dfU = dfU[:, [:T, :u1, :u2, :u3, :dT, :du1, :du2, :du3]]
# dfU = add_diff(DataFrame(eofed.U[:, 1:3], ["u$j" for j in 1:3]); method = :TVD)[1:(end-1), :]
# dfU = dfU[:, [:u1, :u2, :u3, :du1, :du2, :du3]]

vrbl = reverse(half(names(dfU)))
cnfg = cook(vrbl; poly = 0:2, trig = [2])[[1,3,4,5,10,11,12,13,14,15,16,17], :]
# cnfg = cook(vrbl; trig = 1:5)
# cnfg = cook(vrbl; poly = 0:2)

f = SINDy(dfU, vrbl, cnfg; λ = 1e-4); f |> print

"""''''''''''''''''''''''''''''''''''''''''''''''''''

                        fitting

''''''''''''''''''''''''''''''''''''''''''''''''''"""
# anim = @animate for tk0 = 1:10:1350
#     print("\r tk0 = $tk0 / 1350")
tk0 = 1
# V = ssolve(f, dfU[[tk0], last(vrbl)], dfU.T[tk0]:1e-2:(dfU.T[tk0]+120))[1:100:end, 2:end]
# V = ssolve(f, dfU[[tk0], last(vrbl)], dfU.T[tk0]:1e-2:nrow(dfU))[1:100:end, 2:end]
V = ssolve(f, dfU[[tk0], last(vrbl)], 0:1e-2:nrow(dfU))[1:100:end, 2:end]
ylabel = [L"T", L"u_1", L"u_2", L"u_3"]
U = dfU[:, last(vrbl)]
plt_ = [plot() for j in axes(U, 2)]
for j in eachindex(plt_)
    plot!(plt_[j], U.T, U[:, j], label = "EOF", color = :black, alpha = 0.5, lw = 2, ylabel = ylabel[j])
    plot!(plt_[j], V.T, V[:, j], label = "SINDy+EOF", color = :red, alpha = 0.5, lw = 2)
end
plot(plt_..., layout = (:, 1), legend = :none, size = [600, 600], xticks = 0:1:10)
# end
@time gif(anim, "SINDy+EOF_fitting.gif", fps = 60);

"""''''''''''''''''''''''''''''''''''''''''''''''''''

                    limit cycle

''''''''''''''''''''''''''''''''''''''''''''''''''"""
V = ssolve(f, dfU[[1], last(vrbl)], 0000:1e-2:9000)
V = ssolve(f, dfU[[1], last(vrbl)], 2000:1e-2:3000)
plot(V.u1, V.u2, V.u3, color = :red, legend = :none, size = [400, 400], label = :none, xlabel = ylabel[2], ylabel = ylabel[3], zlabel = ylabel[4])
scatter!(V.u1[[end]], V.u2[[end]], V.u3[[end]], color = :red, msc = :white)

plot(U.u1, U.u2, U.u3, color = :black, )
plot!(V.u1, V.u2, V.u3, color = :red, α = 0.5)

plot(
    plot(V.u1),
    plot(V.u2),
    plot(V.u3)
)
"""''''''''''''''''''''''''''''''''''''''''''''''''''

                performance evaluation

''''''''''''''''''''''''''''''''''''''''''''''''''"""
year0 = 2023
test = Matrix(nino34[(Date(year0) .≤ T), id_target])
test = test .- μ
_test = DataFrame(eof2U(test, 3), ["u1", "u2", "u3"])
_test = [DataFrame(T = (1:nrow(_test))/365) _test]
tend = 120
# Uhat = ssolve(f, collect(dfU[end, last(vrbl)]), 0:1e-2:tend)[1:100:end, 3:end];
# Xhat = rcv(eof2Z(Uhat, 3)[1, :])
# heatmap(Y, X, to2(Xhat))
resultnames = "lead" .* string.(0:tend)
resultSINDy = DataFrame([zeros(length(T[Date(year0) .≤ T])) for _ in resultnames], resultnames)
resultSINDy = [DataFrame(date = T[Date(year0) .≤ T]) resultSINDy]
@showprogress @threads for t0 = eachindex(T[Date(year0) .≤ T])
    uhat = ssolve(f, _test[[t0], :], 0:1e-2:tend)[1:100:end, 3:end]
    actl = test[t0 .+ (0:tend), :]
    zhat = eof2Z(Matrix(uhat), 3)
    resultSINDy[t0, 2:end] = mae(actl, zhat, dims = 2)
end
resultSINDy = resultSINDy[.!iszero.(resultSINDy[:, 2]), :]
CSV.write("resultSINDy.csv", resultSINDy)

resultSINDy = CSV.read("resultSINDy3years.csv", DataFrame)
# surface(0:120, _test.T[1:1121], Matrix(resultSIDy[:, 2:end]), xlabel = "lead", ylabel = "ic", size = [400, 400])
plot(legend = true, xlabel = "lead", ylabel = "average MAE", xticks = 0:30:180, ylims = [0.2, 1.2])
plot!(0:tend, mean.(eachcol(resultSINDy[Year(2023) .≤ Year.(resultSINDy.date) .< Year(2024), Not(1)])), lw = 2, α = 0.5, color = 1, label = "2023")
plot!(0:tend, mean.(eachcol(resultSINDy[Year(2024) .≤ Year.(resultSINDy.date) .< Year(2025), Not(1)])), lw = 2, α = 0.5, color = 2, label = "2024")
plot!(0:tend, mean.(eachcol(resultSINDy[Year(2025) .≤ Year.(resultSINDy.date) .< Year(2026), Not(1)])), lw = 2, α = 0.5, color = 3, label = "2025")
plot!(0:tend, mean.(eachcol(resultSINDy[Year(2023) .≤ Year.(resultSINDy.date) .< Year(2026), Not(1)])), lw = 2, α = 0.5, color = :black, label = "2023-2026")


"""''''''''''''''''''''''''''''''''''''''''''''''''''

                    equation analysis

''''''''''''''''''''''''''''''''''''''''''''''''''"""
f_ = [SINDy(dfU[Not(i), :], vrbl, cnfg; λ = 1e-4) for i in 1:nrow(dfU)]
m_ = [f.matrix for f in f_]
plot(
    heatmap(mean(m_), xmirror = true, yflip = true, colorbar = true),
    heatmap(std(m_), xmirror = true, yflip = true, colorbar = true),
)

"""''''''''''''''''''''''''''''''''''''''''''''''''''

                    embeded points

''''''''''''''''''''''''''''''''''''''''''''''''''"""

plot(
    scatter(_test.u1, _test.u2, _test.u3, msw = 0, alpha = 0.5, color = ceil.(Int64, _test.T)),
    scatter(_test.u1, _test.u2, msw = 0, alpha = 0.5, color = ceil.(Int64, _test.T)),
    scatter(_test.u2, _test.u3, msw = 0, alpha = 0.5, color = ceil.(Int64, _test.T)),
    scatter(_test.u1, _test.u3, msw = 0, alpha = 0.5, color = ceil.(Int64, _test.T)),
    layout = (2, 2), size = [800, 600]
)


heatmapreshape(collect(nino34m[1, :]), 251, :)