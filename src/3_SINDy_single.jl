ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,6]])
include("0_utils.jl")
include("1_datacall.jl")

id_ovserver = []
id_target = setdiff(axes(nino34, 2), [id_missing; id_ovserver])
rcv(x) = recover(x, (id_target, []))

trng = Matrix(nino34[(Date(2019) .≤ T .< Date(2023)), id_target])

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


using FFTW

foo = fft(nino34[:, 150000])
top10 = sort(norm.(foo[1:length(half(foo)[1])]), rev = true)[10]
foo[norm.(foo) .< top10] .= 0
bar = real.(ifft(foo))
plot(nino34[:, 150000])
plot!(bar)

mean(abs, nino34[:, 150000] - bar)

toy = add_diff(DataFrame(x = bar), method = :TVD); rename!(toy, ["y", "x"])
toy = add_diff(toy, method = :TVD)

plot(toy.x, toy.y, toy.dy)
plot(
    plot(toy.x),
    plot(toy.y, yticks = [0]),
    plot(toy.dy, yticks = [0]),
    layout = (:, 1),
    size = [600, 600],
)
vrbl = (["dy", "dx"], ["y", "x"])
cnfg = cook(vrbl; poly = 0:1)
f = SINDy(toy, vrbl, cnfg; λ = 1e-4); f |> print
out = ssolve(f, toy[[1], :], 0:1e-2:nrow(toy))[1:100:(end-1), :]

plot(toy.x)
plot!(out.x, color = :red, α = 0.5)

mean(abs, (toy.x - out.x)[1:365, :])

to2(nino34[[210], :]) |> heatmap
heatmap(to2(nino34[[1], :])[1:10:end, 1:5:end], size = [500, 100])
size(to2(nino34[[1], :])[1:10:end, 1:5:end])

ismissing.(to2(nino34[[1], :])[1:10:end, 1:5:end]) |> count