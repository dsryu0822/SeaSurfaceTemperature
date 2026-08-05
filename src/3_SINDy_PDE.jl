ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,6]])
include("0_utils.jl")

@time JLD2.@load "nino34tuv.jld2"

# nino34T = [DataFrame(date=D) DataFrame(reshape(T, prod(size(T[:,:,1])), :)', ["T$k" for k in 1:3125])]; CSV.write("G:/seasurface/nino34T.csv", nino34T)
# nino34U = [DataFrame(date=D) DataFrame(reshape(U, prod(size(U[:,:,1])), :)', ["U$k" for k in 1:3125])]; CSV.write("G:/seasurface/nino34U.csv", nino34U)
# nino34V = [DataFrame(date=D) DataFrame(reshape(V, prod(size(V[:,:,1])), :)', ["V$k" for k in 1:3125])]; CSV.write("G:/seasurface/nino34V.csv", nino34V)

_dT = diff(T, dims = 3); _dU = diff(U, dims = 3); _dV = diff(V, dims = 3);
_T = T[:, :, 1:end-1]; _U = U[:, :, 1:end-1]; _V = V[:, :, 1:end-1];

_Tx = circshift(_T, (0, -1, 0)) - circshift(_T, (0, 1, 0))
_Ty = circshift(_T, (-1, 0, 0)) - circshift(_T, (1, 0, 0))
_Txx = circshift(_T, (0, -1, 0)) - 2*_T + circshift(_T, (0, 1, 0))
_Tyy = circshift(_T, (-1, 0, 0)) - 2*_T + circshift(_T, (1, 0, 0))
_Ux = circshift(_U, (0, -1, 0)) - circshift(_U, (0, 1, 0))
_Uy = circshift(_U, (-1, 0, 0)) - circshift(_U, (1, 0, 0))
_Uxx = circshift(_U, (0, -1, 0)) - 2*_U + circshift(_U, (0, 1, 0))
_Uyy = circshift(_U, (-1, 0, 0)) - 2*_U + circshift(_U, (1, 0, 0))
_Vx = circshift(_V, (0, -1, 0)) - circshift(_V, (0, 1, 0))
_Vy = circshift(_V, (-1, 0, 0)) - circshift(_V, (1, 0, 0))
_Vxx = circshift(_V, (0, -1, 0)) - 2*_V + circshift(_V, (0, 1, 0))
_Vyy = circshift(_V, (-1, 0, 0)) - 2*_V + circshift(_V, (1, 0, 0))
for W in [_Tx, _Ty, _Txx, _Tyy, _Ux, _Uy, _Uxx, _Uyy, _Vx, _Vy, _Vxx, _Vyy]
    W[1, :, :] .= W[2, :, :]
    W[:, 1, :] .= W[:, 2, :]
    W[end, :, :] .= W[end-1, :, :]
    W[:, end, :] .= W[:, end-1, :]
end

data = DataFrame(dT = vec(_dT[2:(end-1),2:(end-1),:]), dU = vec(_dU[2:(end-1),2:(end-1),:]), dV = vec(_dV[2:(end-1),2:(end-1),:]), T = vec(_T[2:(end-1),2:(end-1),:]), U = vec(_U[2:(end-1),2:(end-1),:]), V = vec(_V[2:(end-1),2:(end-1),:]), Tx = vec(_Tx[2:(end-1),2:(end-1),:]), Ty = vec(_Ty[2:(end-1),2:(end-1),:]), Txx = vec(_Txx[2:(end-1),2:(end-1),:]), Tyy = vec(_Tyy[2:(end-1),2:(end-1),:]), Ux = vec(_Ux[2:(end-1),2:(end-1),:]), Uy = vec(_Uy[2:(end-1),2:(end-1),:]), Uxx = vec(_Uxx[2:(end-1),2:(end-1),:]), Uyy = vec(_Uyy[2:(end-1),2:(end-1),:]), Vx = vec(_Vx[2:(end-1),2:(end-1),:]), Vy = vec(_Vy[2:(end-1),2:(end-1),:]), Vxx = vec(_Vxx[2:(end-1),2:(end-1),:]), Vyy = vec(_Vyy[2:(end-1),2:(end-1),:]))

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
