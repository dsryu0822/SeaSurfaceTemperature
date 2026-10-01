ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,6]])
include("0_utils.jl")
@time JLD2.@load "G:/seasurface/nino34TUV.jld2"

∂x(M) = (circshift(M, (0, -1)) - circshift(M, (0, 1)))
∂y(M) = (circshift(M, (-1, 0)) - circshift(M, (1, 0)))
∂xx(M) = (circshift(M, (0, -1)) - 2M + circshift(M, (0, 1)))
∂yy(M) = (circshift(M, (-1, 0)) - 2M + circshift(M, (1, 0)))

function take(boundary = false; i = 1:41, j = 1:201, k = 1:2207)
    bits = zeros(Bool, 41, 201, 2207)
    bits[i, j, k] .= true
    if boundary
        bits[[1, end], :, :] .= true
        bits[:, [1, end], :] .= true
    else
        bits[[1, end], :, :] .= false
        bits[:, [1, end], :] .= false
    end
    return DataFrame(dT = vec(_dT[bits]), dU = vec(_dU[bits]), dV = vec(_dV[bits]), T = vec(_T[bits]), U = vec(_U[bits]), V = vec(_V[bits]), Tx = vec(_Tx[bits]), Ty = vec(_Ty[bits]), Txx = vec(_Txx[bits]), Tyy = vec(_Tyy[bits]), Ux = vec(_Ux[bits]), Uy = vec(_Uy[bits]), Uxx = vec(_Uxx[bits]), Uyy = vec(_Uyy[bits]), Vx = vec(_Vx[bits]), Vy = vec(_Vy[bits]), Vxx = vec(_Vxx[bits]), Vyy = vec(_Vyy[bits]))
end


# T only
vrblT = (["dT"], ["T", "U", "V", "Tx", "Ty", "Txx", "Tyy"])
cnfgT = cook(vrblT; poly = 0:2)[[1,18,24,7,8], :]
@time fT = SINDy(data, vrblT, cnfgT); fT |> print

# U only
vrblU = (["dU"], ["U", "V", "Ux", "Uy", "Uxx", "Uyy"])
cnfgU = cook(vrblU; poly = 0:2)[[1,10,16,6,7,3], :]
@time fU = SINDy(data, vrblU, cnfgU); fU |> print

# V only
vrblV = (["dV"], ["U", "V", "Vx", "Vy", "Vxx", "Vyy"])
cnfgV = cook(vrblV; poly = 0:2)[[1,10,16,6,7,2], :]
@time fV = SINDy(data, vrblV, cnfgV); fV |> print


define(fT, sigdigits = 5) |> print
define(fU, sigdigits = 5) |> print
define(fV, sigdigits = 5) |> print
function simulate(t0 = 1)
    mae_ = []
    Δ = 1e-2; tend = 120
    T0 = _T[:, :, t0]; U0 = _U[:, :, t0]; V0 = _V[:, :, t0]
    dirichlet  = zeros(Bool, size(T0)...); dirichlet[[1, end], :] .= true; dirichlet[:, [1, end]] .= true;
    T = _T[:, :, t0]; U = _U[:, :, t0]; V = _V[:, :, t0]; Tx = _Tx[:, :, t0]; Ty = _Ty[:, :, t0]; Txx = _Txx[:, :, t0]; Tyy = _Tyy[:, :, t0]; Ux = _Ux[:, :, t0]; Uy = _Uy[:, :, t0]; Uxx = _Uxx[:, :, t0]; Uyy = _Uyy[:, :, t0]; Vx = _Vx[:, :, t0]; Vy = _Vy[:, :, t0]; Vxx = _Vxx[:, :, t0]; Vyy = _Vyy[:, :, t0];
    anime = @animate for t in 1:tend
        print("\rt = $t / $tend")
        for _t in range(0, 1, step = Δ)
            T .+= Δ*(0.0093528 .+ ( - 0.56712U.*Tx - 0.37157V.*Ty + 0.31923Txx + 0.13743Tyy))
            U .+= Δ*(0.0026584 .+ ( - 0.36422U.*Ux - 0.15823V.*Uy + 0.35914Uxx + 0.064746Uyy + 0.0077646V))
            V .+= Δ*(7.0035e-5 .+ ( - 0.4961U.*Vx - 0.25306V.*Vy + 0.17146Vxx + 0.2651Vyy - 0.0082463U))

            T[dirichlet] .= T0[dirichlet]
            U[dirichlet] .= U0[dirichlet]
            V[dirichlet] .= V0[dirichlet]

            Tx = circshift(T, (0, -1)) - circshift(T, (0, 1))
            Ty = circshift(T, (-1, 0)) - circshift(T, (1, 0))
            Txx = circshift(T, (0, -1)) - 2*T + circshift(T, (0, 1))
            Tyy = circshift(T, (-1, 0)) - 2*T + circshift(T, (1, 0))
            Ux = circshift(U, (0, -1)) - circshift(U, (0, 1))
            Uy = circshift(U, (-1, 0)) - circshift(U, (1, 0))
            Uxx = circshift(U, (0, -1)) - 2*U + circshift(U, (0, 1))
            Uyy = circshift(U, (-1, 0)) - 2*U + circshift(U, (1, 0))
            Vx = circshift(V, (0, -1)) - circshift(V, (0, 1))
            Vy = circshift(V, (-1, 0)) - circshift(V, (1, 0))
            Vxx = circshift(V, (0, -1)) - 2*V + circshift(V, (0, 1))
            Vyy = circshift(V, (-1, 0)) - 2*V + circshift(V, (1, 0))
        end
        _mae = mae(T[2:(end-1), 2:(end-1)], _T[2:(end-1), 2:(end-1), t0 + t])
        push!(mae_, _mae)
        plot(
            heatmap(_T[:, :, t0 + t], legend = :none, color = RdYlBu_r),
            heatmap(T, legend = :none, color = RdYlBu_r),
            plot(mae_, legend = :none, color = :black, xlims = [0, tend], ylims = [0, 1]),
            clims = (19, 31),
            layout = (:, 1), size = [600, 400], plot_title = "t = $t0 + $t ($(D[t]))"
        )
    end
    println()
    gif(anime, "temp1.gif", fps = 10);
end

simulate()
"""''''''''''''''''''''''''''''''''''''''''''''''''''

                    Coriolis effect

''''''''''''''''''''''''''''''''''''''''''''''''''"""


take(j = [2])
data = shuffle(take())[1:1000, :]
data = shuffle(take())[1:1000, :]
fT = SINDy(data, vrblT, cnfgT); fT |> print
fU = SINDy(data, vrblU, cnfgU); fU |> print
fV = SINDy(data, vrblV, cnfgV); fV |> print

coriU = []
coriV = []
for y in 2:40
    taken = take(i = [y])
    fT = SINDy(taken, vrblT, cnfgT); # fT |> print
    fU = SINDy(taken, vrblU, cnfgU); fU |> print
    fV = SINDy(taken, vrblV, cnfgV); # fV |> print
    push!(coriU, fU.matrix[end])
    push!(coriV, fV.matrix[end])
end
plot(plot(Y[2:40], coriU, ylabel = "f"), plot(Y[2:40], coriV), xlabel = "y")

