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



"""''''''''''''''''''''''''''''''''''''''''''''''''''

                        PINN

''''''''''''''''''''''''''''''''''''''''''''''''''"""

# _T, _U, _V, X, Y가 로딩된 상태에서 이 섹션을 실행한다.
using Flux, Random, Statistics

t0 = 1
tend = 5.0                      # 예측 기간 (일)
epochs = 1000
v, κ = 100.0, 100.0              # 유효 점성/열확산계수 (m²/s), 실험용 값
Random.seed!(42)

T0, U0, V0 = _T[:,:,t0], _U[:,:,t0], _V[:,:,t0]
ny, nx = size(T0)
Lx = 6_371_000 * cosd(mean(Y)) * deg2rad(last(X)-first(X))
Ly = 6_371_000 * deg2rad(last(Y)-first(Y))
ts = tend * 86400
us = max(sqrt(mean(abs2,U0) + mean(abs2,V0)), 0.01)
Ts = max(std(T0), 0.1)
offset05 = [0., 0., mean(T0), 0.]
scale05 = [us, us, Ts, us^2]      # 출력: u, v, T, π=p/ρ₀
ax, ay = ts*us/Lx, ts*us/Ly
vx, vy, κx, κy = ts*v/Lx^2, ts*v/Ly^2, ts*κ/Lx^2, ts*κ/Ly^2

# 좌표 (시간, x, y)를 [0,1]로 정규화. 배열은 (y,x).
grid05 = vcat(zeros(1,ny*nx),
    reshape([(x-first(X))/(last(X)-first(X)) for y in Y,x in X],1,:),
    reshape([(y-first(Y))/(last(Y)-first(Y)) for y in Y,x in X],1,:))
q0 = ([vec(U0)'; vec(V0)'; vec(T0)'] .- offset05[1:3]) ./ scale05[1:3]
@assert all(isfinite,q0) "초기장에 missing/NaN이 없는 영역을 선택하세요"
edge05 = findall(vec([i in (1,ny) || j in (1,nx) for i in 1:ny,j in 1:nx]))
ic05 = rand(1:ny*nx,512)
bc05 = rand(edge05,512)
net05 = Flux.f64(Chain(Dense(3=>32,tanh), Dense(32=>32,tanh), Dense(32=>4)))

# tanh의 연쇄법칙으로 1·2차 미분 계산.
function derivatives05(m,z)
    h = 2z .- 1
    dt = repeat([2.,0.,0.],1,size(z,2))
    dx = repeat([0.,2.,0.],1,size(z,2))
    dy = repeat([0.,0.,2.],1,size(z,2))
    dxx, dyy = zero(h), zero(h)
    for (k,l) in enumerate(m.layers)
        a = l.weight*h .+ l.bias
        at, ax_, ay_ = l.weight*dt, l.weight*dx, l.weight*dy
        axx, ayy = l.weight*dxx, l.weight*dyy
        k == length(m.layers) && return (a,at,ax_,ay_,axx,ayy)
        h = tanh.(a)
        g, gg = 1 .- h.^2, -2h .* (1 .- h.^2)
        dt, dx, dy = g.*at, g.*ax_, g.*ay_
        dxx, dyy = gg.*ax_.^2 .+ g.*axx, gg.*ay_.^2 .+ g.*ayy
    end
end

function loss05(m,z,zbc)
    q, qt, qx, qy, qxx, qyy = derivatives05(m,z)
    u, v = q[1:1,:], q[2:2,:]
    # z방향을 제외한 운동량 2식·열수송식·비압축성 조건.
    ru = qt[1:1,:] + ax*u.*qx[1:1,:] + ay*v.*qy[1:1,:] + ax*qx[4:4,:] - vx*qxx[1:1,:] - vy*qyy[1:1,:]
    rv = qt[2:2,:] + ax*u.*qx[2:2,:] + ay*v.*qy[2:2,:] + ay*qy[4:4,:] - vx*qxx[2:2,:] - vy*qyy[2:2,:]
    rT = qt[3:3,:] + ax*u.*qx[3:3,:] + ay*v.*qy[3:3,:] - κx*qxx[3:3,:] - κy*qyy[3:3,:]
    div = qx[1:1,:] + (Lx/Ly)*qy[2:2,:]
    # 초기조건 + 초기장의 경계값을 모든 시점에 고정하는 Dirichlet 조건 (loss로 근사).
    ic = m(2grid05[:,ic05] .- 1)[1:3,:] - q0[:,ic05]
    bc = m(2zbc .- 1)[1:3,:] - q0[:,bc05]
    gauge = m(2vcat(z[1:1,:],fill(0.5,2,size(z,2))) .- 1)[4:4,:]
    return mean(abs2,ru) + mean(abs2,rv) + mean(abs2,rT) + mean(abs2,div) +
        10mean(abs2,ic) + 10mean(abs2,bc) + 0.1mean(abs2,gauge)
end

opt05 = Flux.setup(Flux.Adam(1e-3),net05)
loss_history05 = Float64[]
for epoch in 1:90000
    z = rand(3,256)
    zbc = vcat(rand(1,length(bc05)),grid05[2:3,bc05])
    loss, grads = Flux.withgradient(m -> loss05(m,z,zbc),net05)
    isfinite(loss) || error("loss가 발산했습니다. 학습률이나 tend를 줄이세요")
    Flux.update!(opt05,net05,grads[1])
    push!(loss_history05,loss)
    epoch % 100 == 0 && println("epoch=$epoch, loss=$loss")
end

# tend일 후의 장을 원래 격자로 복원하고 표시.
zpred05 = copy(grid05); zpred05[1,:] .= 1
qpred05 = net05(2zpred05 .- 1) .* scale05 .+ offset05
Upinn = reshape(qpred05[1,:],ny,nx)
Vpinn = reshape(qpred05[2,:],ny,nx)
Tpinn = reshape(qpred05[3,:],ny,nx)
plot(heatmap(T0,title="Initial T"), heatmap(Tpinn,title="PINN T, +$tend days"),
    plot(loss_history05,scale=:log10,title="loss"), layout=(3,1),size=(600,600))
