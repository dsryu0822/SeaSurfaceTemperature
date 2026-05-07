include("0_utils.jl")
# JLD2.@save "G:/seasurface/nino3.4/data.jld2" data T X Y
@time JLD2.@load "G:/seasurface/nino3.4/data.jld2"
include.(readdir("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core", join=true))

to2(x) = reshape(collect(x), length(Y), length(X))
datasplit(array, bits) = (array[bits,:], array[.!bits,:])
recover(matrices, indices) = matrices[:, sortperm(indices)]

id_missing = findall((eltype.(eachcol(data))) .== Missing)
function recoverH(matrices, indice)
    void = fill(missing, size(matrices, 1), length(id_missing))
    return recover([matrices;; void], [indice; id_missing])
end
id_ovserver = []
id_target = setdiff(axes(data, 2), [id_missing; id_ovserver])

# 가장 뒷줄에 t = 1, ... 추가
_data = [data DataFrame(t = axes(data, 1))]
trng, test = datasplit(_data, T .< Date(2023, 1, 1));

@time eofed = eof(Matrix(trng)[:, [id_target; end]])
eof_transform(xx, nu) = xx / (diagm(eofed.Σ[1:nu]) * eofed.V[:, 1:nu]')
eof_transform(Matrix(test[1:2, [id_target; end]]), 3)
plot(cumsum(eofed.Σ ./ sum(eofed.Σ)), xscale = :log10, label = :none, color = :black)

# N = 3까지 사용
U = eofed.U[:, 1:3]
dfU = add_diff(DataFrame(U, :auto); method = :TVD)
vrbl = half(names(dfU))

plot([plot(u) for u in eachcol(U)]...,layout = (:, 1), legend = :none)

f = SINDy(dfU, vrbl, cook(last(vrbl); poly = 0:2)); f |> print
tend = 300
@time V = solve(f, collect(dfU[end, last(vrbl)]), 0:1e-2:tend)[1:100:end, :][2:end, :];
futureU = eof_transform(Matrix(test[1:300, [id_target; end]]), 3)
plt_l_ = []
for j in axes(V, 2)
    U = futureU
    plt_l = plot(U[1:tend, j], ylabel = "EOF$j", label = "EOF")
    plot!(plt_l, V[:, j], label = "SINDy")
    push!(plt_l_, plt_l)
end
plot(plt_l_..., layout = (:, 1), legend = :none, size = [600, 200length(plt_l_)])

foo = recoverH(eofed(V)[:, Not(end)], id_target)
heatmap(Y, X, to2(foo[30, :])', size = [800, 200])

actl = Matrix(test[1:tend, id_target]);
prdt = Matrix(foo[1:end, id_target]);

plot(sqrt.(mean.(abs2, eachrow(actl - prdt))), label = :none, color = :black)

# 아래부터는 생각보다 EOF로 보는 손실이 큰 것 확인

plot(sqrt.(mean.(abs2, eachrow(eofed(eofed.U[:, 1:3])[:, Not(end)] - Matrix(trng[:, id_target])))), color = :black, label = :none)

(eofed.Σ ./ sum(eofed.Σ)) |> cumsum

# N = 5까지 사용
dfU = add_diff(DataFrame(eofed.U[:, 1:5], :auto); method = :TVD)
vrbl = half(names(dfU))
f = SINDy(dfU, vrbl, cook(last(vrbl); poly = 0:2), λ = 1e-3); f |> print
tend = 100
@time V = solve(f, collect(dfU[end, last(vrbl)]), 0:1e-1:tend)[1:10:end, :][2:end, :]
futureU = eof_transform(Matrix(test[1:300, [id_target; end]]), 5)
plt_l_ = []
for j in axes(futureU, 2)
    U = futureU
    plt_l = plot(U[1:tend, j], ylabel = "EOF$j", label = "EOF")
    plot!(plt_l, V[:, j], label = "SINDy")
    push!(plt_l_, plt_l)
end
plot(plt_l_..., layout = (:, 1), legend = :none, size = [600, 200length(plt_l_)])
