ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,6]])
include("0_utils.jl")

@time JLD2.@load "G:/seasurface/nino34tuv.jld2"

# nino34T = [DataFrame(date=D) DataFrame(reshape(T, prod(size(T[:,:,1])), :)', ["T$k" for k in 1:3125])]; CSV.write("G:/seasurface/nino34T.csv", nino34T)
# nino34U = [DataFrame(date=D) DataFrame(reshape(U, prod(size(U[:,:,1])), :)', ["U$k" for k in 1:3125])]; CSV.write("G:/seasurface/nino34U.csv", nino34U)
# nino34V = [DataFrame(date=D) DataFrame(reshape(V, prod(size(V[:,:,1])), :)', ["V$k" for k in 1:3125])]; CSV.write("G:/seasurface/nino34V.csv", nino34V)

nino34T = CSV.read("G:/seasurface/nino34T.csv", DataFrame)

trng = Matrix(nino34T[:, 2:end])

@time eofed = eof(trng)
eof2Z(uu, nu) = uu * (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
eof2U(zz, nu) = zz / (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
plot(cumsum(eofed.Σ ./ sum(eofed.Σ)), xscale = :log10, label = :none, color = :black)

data = data[:, [:u1, :u2, :u3, :du1, :du2, :du3]]
vrbl = reverse(half(names(data)))
cnfg = cook(vrbl; poly = 0:2)

f = SINDy(data, vrbl, cnfg; λ = 1e-4); f |> print

tk0 = 1
pU = ssolve(f, data[[tk0], last(vrbl)], 0:1e-2:nrow(data))[1:100:end, 2:end]
aU = data[:, last(vrbl)]
plt_ = [plot() for j in axes(aU, 2)]
for j in eachindex(plt_)
    plot!(plt_[j], aU[:, j], label = "EOF", color = :black, alpha = 0.5, lw = 2)
    plot!(plt_[j], pU[:, j], label = "SINDy+EOF", color = :red, alpha = 0.5, lw = 2)
end
plot(plt_..., layout = (:, 1))


year0 = 2026
tend = 30
test = Matrix(trng[(Date(year0) .≤ D), :])
_test = DataFrame(eof2U(test, 3), ["u1", "u2", "u3"])

resultnames = "lead" .* string.(0:tend)
resultSINDy = DataFrame([zeros(length(D[Date(year0) .≤ D])) for _ in resultnames], resultnames)
resultSINDy = [DataFrame(date = D[Date(year0) .≤ D]) resultSINDy]
@showprogress @threads for t0 = eachindex(D[Date(year0) .≤ D])
    uhat = ssolve(f, _test[[t0], :], 0:1e-2:tend)[1:100:end, Not(:t)]
    actl = test[t0 .+ (0:tend), :]
    zhat = eof2Z(Matrix(uhat), 3)
    resultSINDy[t0, 2:end] = mae(actl, zhat, dims = 2)
end
resultSINDy = resultSINDy[.!iszero.(resultSINDy[:, 2]), :]

plot(mean(Matrix(resultSINDy[:, 2:end]), dims = 1)[:])