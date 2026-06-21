include("0_utils.jl")
# JLD2.@save "G:/seasurface/nino3.4/data.jld2" data T X Y
@time JLD2.@load "G:/seasurface/nino3.4/data.jld2"
include.(readdir("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core", join=true))

to2(x) = reshape(collect(x), length(Y), length(X))
datasplit(array, bits) = (array[bits,:], array[.!bits,:])
recover(matrices, indices) = matrices[:, sortperm(indices)]

id_missing = findall((eltype.(eachcol(data))) .== Missing)
function recoverH(matrices, indice)
    matrices = Matrix(matrices)
    void = fill(missing, size(matrices, 1), length(id_missing))
    return recover([matrices;; void], [indice; id_missing])
end
id_ovserver = []
id_target = setdiff(axes(data, 2), [id_missing; id_ovserver])

trng, test = datasplit(data, T .< Date(2023, 1, 1));

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
cnfg = cook(last(vrbl); poly = 0:2, trig = [2])[[1,3,4,5,10,11,12,13,14,15,16,17], :]
f = SINDy(dfU, vrbl, cnfg; λ = 1e-5); f |> print

tend = 100
# @time Uhat = solve(f, collect(dfU[end, last(vrbl)]), 0:1e-2:tend)[1:100:end, 2:end];
# Xhat = recoverH(eof2Z(Uhat, 3), id_target)
# stack(to2.(eachrow(Xhat)), dims = 3)
# heatmap(Y, X, to2(Xhat[20, :])')
resultnames = ["t0"; "MAE" .* string.(0:100)]
resultSINDy = DataFrame([[] for _ in resultnames], resultnames)
@showprogress for t0 = 1:365
    u0 = [t0/365; vec(eof2U(collect(test[t0, id_target])', 3))]
    uhat = solve(f, u0, 0:1e-2:tend)[1:100:end, 2:end]
    Xhat = recoverH(eof2Z(uhat, 3), id_target)
    A2D = stack(to2.(eachrow(recoverH(test[t0 .+ (0:tend), id_target], id_target))), dims = 3)
    P2D = stack(to2.(eachrow(Xhat)), dims = 3)
    E2D = A2D - P2D

    errtend = [mean(abs, skipmissing(e2d)) for e2d in eachslice(E2D, dims = 3)]
    push!(resultSINDy, [t0; errtend])
    clims = extrema(skipmissing([A2D P2D]))
    plot(
        heatmap(A2D[:,:,30]'; clims, title = "Actual t0+30"),
        heatmap(P2D[:,:,30]'; clims, title = "Predicted t0+30"),
        heatmap(E2D[:,:,30]'; color = :balance),
        plot(0:100, errtend, color = :black, label = "MAE", ylims = [0, 1]),
        size = [600, 600], layout = (:, 1), plot_title = "t0 = $t0"
    )
    png("SINDy+EOF_t0_$(lpad(t0, 3, '0')).png")
end
# CSV.write("resultSINDy.csv", resultSINDy)
plot(0:100, mean.(eachcol(resultSINDy))[2:end], xlabel = "lead", color = :black, label = "average MAE", xticks = 0:30:90, ylims = [0, 1])
