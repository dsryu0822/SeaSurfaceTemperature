ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,6]])
include("0_utils.jl")
# include("1_datacall.jl")
@time "✅ nino3.4 jld2" JLD2.@load "../nino3.4/datam.jld2"

id_ovserver = []
id_target = setdiff(axes(nino34, 2), [id_missing; id_ovserver])
rcv(x) = recover(x, (id_target, []))

trng = nino34m[1:36, id_target]
# test = nino34m[73:end, id_target]

plot(nino34m.x1)

@time eofed = eof(Matrix(trng))
eof2Z(uu, nu) = uu * (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
eof2U(zz, nu) = zz / (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
plot(cumsum(eofed.Σ ./ sum(eofed.Σ)), xscale = :log10, label = :none, color = :black)

U = DataFrame(u = eofed.U[:, 1])
dU = rename(add_diff(U, method = :FDM), ["x", "y"])
ddU = rename(add_diff(dU, method = :FDM), ["dx", "dy", "x", "y"])
plot(plot.(eachcol(ddU))..., layout = (:, 1))

vrbl = half(names(ddU))
cnfg = cook(vrbl; trig = 0:2)

f = SINDy(ddU, vrbl, cnfg; λ = 1e-4); f |> print
V = ssolve(f, collect(ddU[1, last(vrbl)]), 0:1e-2:nrow(U))[1:100:end, :]
plt_ = [plot() for j in axes(U, 2)]
for j in eachindex(plt_)
    plot!(plt_[j], U[:, j], label = "EOF", color = :black)
    plot!(plt_[j], V[:, j+2], label = "SINDy+EOF", color = :red)
end
plot(plt_..., layout = (:, 1), legend = :none, size = [600, 200length(plt_)], xticks = 0:365:2000)

import Durbyn as db
import Durbyn.ARIMA as ar

out = db.auto_arima(eofed.U[:, 3], 12)
fc = db.forecast(out, h = 60)
plot(out.y)
plot!([fc.fitted; fc.mean])
stack([circshift(U.u, k) for k in 0:12])

df = [rename(dw(add_diff(U)), ["fdu", "fu"]) up(add_diff(U))]
vrbl = half(names(df))
cnfg = cook(vrbl; poly = 0:2)

f = SINDy(df, vrbl, cnfg; λ = 1e-4); f |> print
predu = [collect(df[1, last(vrbl)])]
for t = 1:69
    push!(predu, f(predu[end]))
end
plot(df.u)
plot!(last.(predu))

u = eofed.U[:, 1]
U = DataFrame(stack([circshift(u, k) for k in 0:12])[13:end, :], :auto)
vrbl = (["x1"], ["x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x13"])
cnfg = cook(vrbl; poly = 0:1)

f = SINDy(U, vrbl, cnfg; λ = 1e-4); f |> print
predu = [collect(U[t, :]) for t in 1:13]
for t = 1:40
    push!(predu, [f(predu[end][2:end]); predu[end][2:end]])
end
plot(U[:, 1])
plot!(first.(predu))
