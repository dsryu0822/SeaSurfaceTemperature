ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,6]])
include("0_utils.jl")
include("1_datacall.jl")

# to2(x) = reshape(collect(x), length(Y), length(X))
# datasplit(array, bits) = (array[bits,:], array[.!bits,:])
# recover(matrices, indices) = matrices[:, sortperm(indices)]

# id_missing = findall((eltype.(eachcol(data))) .== Missing)
# function recoverH(matrices, indice)
#     matrices = Matrix(matrices)
#     void = fill(missing, size(matrices, 1), length(id_missing))
#     return recover([matrices;; void], [indice; id_missing])
# end

# trng = Matrix(nino34[(T .< Date(2024)), :])
# trng = Matrix(nino34[(T .< Date(2025)), :])
trng = Matrix(nino34[(Date(2019) .≤ T .< Date(2025)), :])
test = Matrix(nino34[(Date(2025) .≤ T), id_target])

id_ovserver = []
id_target = setdiff(axes(nino34, 2), [id_missing; id_ovserver])
rcv(x) = recover(x, (id_target, []))

@time eofed = eof(Matrix(trng)[:, id_target])
eof2Z(uu, nu) = uu * (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
eof2U(zz, nu) = zz / (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
plot(cumsum(eofed.Σ ./ sum(eofed.Σ)), xscale = :log10, label = :none, color = :black)

U = [DataFrame(t = axes(trng, 1) ./ 365) DataFrame(eofed.U[:, 1:3], :auto)]
dfU = [
    add_diff(DataFrame(t = axes(trng, 1) ./ 365); method = :FDM) add_diff(DataFrame(eofed.U[:, 1:3], :auto); method = :TVD)[1:(end-1), :]
]
dfU = dfU[:, sortperm(names(dfU))]
vrbl = half(names(dfU))
# plot([plot(u) for u in eachcol(U)]...,layout = (:, 1), legend = :none)
# cnfg = cook(vrbl; poly = 0:3)
cnfg = cook(vrbl; poly = 0:3, trig = [2])[[1,3,4,5,10,11,12,13,14,15,16,17], :]
f = SINDy(dfU, vrbl, cnfg; λ = 1e-3); f |> print
V = ssolve(f, collect(dfU[1, last(vrbl)]), 0:1e-2:nrow(U))[1:100:end, 2:end]
# plot([plot(v) for v in eachcol(V)]...,layout = (:, 1), legend = :none)
plt_ = [plot() for j in axes(V, 2)]
for j in eachindex(plt_)
    plot!(plt_[j], U[:, j], label = "EOF", color = :black)
    plot!(plt_[j], V[:, j], label = "SINDy+EOF", color = :red)
end
plot(plt_..., layout = (:, 1), legend = :none, size = [600, 600], xticks = 0:365:2000)
# plot(U[:, 1], U[:, 2], U[:, 3], color = :black, ticks = [], legend = :none)

# 리미트 사이클
V = ssolve(f, collect(dfU[1, last(vrbl)]), 0:1e-2:10000)[900000:100:end, 2:end]
plot(V[:, 2], V[:, 3], V[:, 4], color = :red)
plot(V[:, 2], color = :red)

tend = 120
# Uhat = ssolve(f, collect(dfU[end, last(vrbl)]), 0:1e-2:tend)[1:100:end, 3:end];
# Xhat = rcv(eof2Z(Uhat, 3)[1, :])
# heatmap(Y, X, to2(Xhat))
resultnames = "lead" .* string.(0:120)
resultSINDy = DataFrame([[] for _ in resultnames], resultnames)
@showprogress for t0 = 1:365
    u0 = [t0/365; vec(eof2U(test[t0, :]', 3))]
    uhat = ssolve(f, u0, 0:1e-2:tend)[1:100:end, 3:end]
    actl = test[t0 .+ (0:tend), :]
    zhat = eof2Z(uhat, 3)
    push!(resultSINDy, mae(actl, zhat, dims = 2)[:])

    # errtend = [mean(abs, skipmissing(e2d)) for e2d in eachslice(E2D, dims = 3)]
    # push!(resultSINDy, [t0; errtend])
    # clims = extrema(skipmissing([A2D P2D]))
    # plot(
    #     heatmap(A2D[:,:,30]'; clims, title = "Actual t0+30"),
    #     heatmap(P2D[:,:,30]'; clims, title = "Predicted t0+30"),
    #     heatmap(E2D[:,:,30]'; color = :balance),
    #     plot(0:100, errtend, color = :black, label = "MAE", ylims = [0, 1]),
    #     size = [600, 600], layout = (:, 1), plot_title = "t0 = $t0"
    # )
    # png("SINDy+EOF_t0_$(lpad(t0, 3, '0')).png")
end
plot(0:tend, mean.(eachcol(resultSINDy)), xlabel = "lead", color = :black, label = "average MAE", xticks = 0:30:90, ylims = [0, 1])
CSV.write("resultSINDy.csv", resultSINDy)


mae(actl, zhat, dims = 2)[:]

mean(abs, actl - zhat, dims = 2)