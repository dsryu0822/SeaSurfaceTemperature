ROOT = ifelse(Sys.KERNEL == :NT, "C:/Users/rmsms/OneDrive/lab/", "/home/chaos")
include.(readdir("$(ROOT)/DataDrivenModel/core", join=true)[[1,2,3,6]])
include("0_utils.jl")
@time JLD2.@load "G:/seasurface/nino34TUV.jld2"

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
    return DataFrame(dT = vec(_dT[bits]), dU = vec(_dU[bits]), dV = vec(_dV[bits]), T = vec(_T[bits]), U = vec(_U[bits]), V = vec(_V[bits]), Tx = vec(_Tx[bits]), Ty = vec(_Ty[bits]), Txx = vec(_Txx[bits]), Tyy = vec(_Tyy[bits]), Ux = vec(_Ux[bits]), Uy = vec(_Uy[bits]), Uxx = vec(_Uxx[bits]), Uyy = vec(_Uyy[bits]), Vx = vec(_Vx[bits]), Vy = vec(_Vy[bits]), Vxx = vec(_Vxx[bits]), Vyy = vec(_Vyy[bits]), y = [Y[id[1]] for id in findall(bits)])
end

"""''''''''''''''''''''''''''''''''''''''''''''''''''

                    NeuralPDE

''''''''''''''''''''''''''''''''''''''''''''''''''"""

# 7.5.0 릴리스 이후의 GPU 수정본. 새 Julia 세션에서 아래 명령을 실행한 뒤 재시작한다.
# import Pkg
# Pkg.add(Pkg.PackageSpec(url="https://github.com/SciML/NeuralPDE.jl", rev="50e4b404557f7a4eeab63bd85919bd4c86b240aa"))

using NeuralPDE, Lux, OptimizationOptimisers, ProgressMeter, Random, Statistics, CUDA
using DomainSets: Interval
import NeuralPDE: Optimization, Zygote

# Julia 1.13 / Zygote 0.7.16: 많은 기울기를 한 CUDA 커널에서 합칠 때의 컴파일 오류를 피한다.
function Zygote.accum(x::CuArray,y::CuArray,z::CuArray,zs::CuArray...)
    return foldl(Zygote.accum,zs; init=Zygote.accum(Zygote.accum(x,y),z))
end

@parameters t x y
@variables u(..) v(..) T(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
Dxx, Dyy = Dx^2, Dy^2

t0 = 6*366; _t = t0-1; t1 = t0+10
Random.seed!(42)

# 관측은 t0-_t:t0까지만 사용. _dQ[:,:,k]는 Q[:,:,k+1]-Q[:,:,k]이므로 마지막 k는 t0-1.
data = take(k = (t0-_t):(t0-1))
# 저장된 이웃 차분을 경도·위도에 대한 미분으로 맞춘다.
dx, dy = X[2]-X[1], Y[2]-Y[1]
data[:,[:Tx,:Ux,:Vx]] ./= 2dx
data[:,[:Ty,:Uy,:Vy]] ./= 2dy
data[:,[:Txx,:Uxx,:Vxx]] ./= dx^2
data[:,[:Tyy,:Uyy,:Vyy]] ./= dy^2

function eatonly(cnfg, terms)
    _cnfg = filter(:term => ∈(terms), cnfg)
    return _cnfg[[findfirst(_cnfg.term .== trm) for trm in terms], :]
end


vrblU = (["dU"], ["U", "V", "Ux", "Uy", "Uxx", "Uyy", "y"])
cnfgU = eatonly(cook(vrblU; poly = 0:2), ["1", "U⋅Ux","Uy⋅V","Uxx","Uyy","V⋅y"])
@time gU = SINDy(data, vrblU, cnfgU); gU |> print
a1, ax, ay, akx, aky, aβ = gU.matrix

vrblV = (["dV"], ["U", "V", "Vx", "Vy", "Vxx", "Vyy", "y"])
cnfgV = eatonly(cook(vrblV; poly = 0:2), ["1", "U⋅Vx", "V⋅Vy", "Vxx", "Vyy", "U⋅y"])
@time gV = SINDy(data, vrblV, cnfgV); gV |> print
b1, bx, by, bkx, bky, bβ = gV.matrix

vrblT = (["dT"], ["T", "U", "V", "Tx", "Ty", "Txx", "Tyy"])
cnfgT = eatonly(cook(vrblT; poly = 0:2), ["1", "Tx⋅U", "Ty⋅V", "Txx", "Tyy"])
@time gT = SINDy(data, vrblT, cnfgT); gT |> print
c1, cx, cy, ckx, cky = gT.matrix

T0, U0, V0 = _T[:,:,t0], _U[:,:,t0], _V[:,:,t0]
ny, nx = size(T0)
grid = [fill(t0, ny*nx)'; reshape([x for y in Y,x in X],1,:); reshape([y for y in Y,x in X],1,:)]
q0 = [vec(U0)'; vec(V0)'; vec(T0)']
id_ic = rand(1:ny*nx, 512)
id_bc = rand(findall(vec([i in (1,ny) || j in (1,nx) for i in 1:ny,j in 1:nx])), 256)
initial, qic, qbc = grid[:,id_ic], q0[:,id_ic], q0[:,id_bc]
boundary = [t0 .+ (t1-t0)*rand(1,length(id_bc)); grid[2:3,id_bc]]
initial, boundary, qic, qbc = cu.((initial,boundary,qic,qbc))

# eqs = [
#     Dt(u(t,x,y)) + u(t,x,y)*Dx(u(t,x,y)) + v(t,x,y)*Dy(u(t,x,y)) ~ (Dxx(u(t,x,y)) + Dyy(u(t,x,y))),
#     Dt(v(t,x,y)) + u(t,x,y)*Dx(v(t,x,y)) + v(t,x,y)*Dy(v(t,x,y)) ~ (Dxx(v(t,x,y)) + Dyy(v(t,x,y))),
#     Dt(T(t,x,y)) + u(t,x,y)*Dx(T(t,x,y)) + v(t,x,y)*Dy(T(t,x,y)) ~ (Dxx(T(t,x,y)) + Dyy(T(t,x,y))),
# ]
eqs = [
    Dt(u(t,x,y)) ~ (a1*u(t,x,y) + ax*u(t,x,y)*Dx(u(t,x,y)) + ay*v(t,x,y)*Dy(u(t,x,y)) + akx*Dxx(u(t,x,y)) + aky*Dyy(u(t,x,y)) + aβ*v(t,x,y)*y),
    Dt(v(t,x,y)) ~ (b1*v(t,x,y) + bx*u(t,x,y)*Dx(v(t,x,y)) + by*v(t,x,y)*Dy(v(t,x,y)) + bkx*Dxx(v(t,x,y)) + bky*Dyy(v(t,x,y)) + bβ*u(t,x,y)*y),
    Dt(T(t,x,y)) ~ (c1*T(t,x,y) + cx*u(t,x,y)*Dx(T(t,x,y)) + cy*v(t,x,y)*Dy(T(t,x,y)) + ckx*Dxx(T(t,x,y)) + cky*Dyy(T(t,x,y)))
]

domains = [t ∈ Interval(Float32(t0),Float32(t1)), x ∈ Interval(Float32.(extrema(X))...), y ∈ Interval(Float32.(extrema(Y))...)]
@named pdesys = PDESystem(eqs, [], domains, [t,x,y], [u(t,x,y),v(t,x,y),T(t,x,y)])

model = Lux.Chain(Lux.Dense(3,100,sin), Lux.Dense(100,100,tanh), Lux.Dense(100,3; init_bias=Returns(Float32.(vec(mean(q0;dims=2))))))
function Loss(_,θ,_)
    return mean(abs2.(NN(initial,θ.u) - qic))
     + mean(abs2.(NN(boundary,θ.u) - qbc))
end

disc = PhysicsInformedNN(model,QuasiRandomTraining(256; resampling=false); init_params=Lux.initialparameters(Random.default_rng(),model), additional_loss=Loss, eval_points=2)
prob = discretize(pdesys, disc; adtype=NeuralPDE.AutoZygote(), u0_constructor=CuArray,p_constructor=x -> x isa Array{Float32} ? CuArray(x) : x)
NN = NeuralPDE.getdefault(first(NeuralPDE.pinn_metadata(prob).networks).NN)

EPOCH = 1000
L_ = Float32[]
prgrs = Progress(EPOCH; color=:blue)
callback = function (state,L)
    if state.iter > length(L_)
        push!(L_,L)
        ProgressMeter.update!(prgrs,state.iter; showvalues=[("epoch", argmin(L_)), ("Loss", minimum(L_))])
    end
    return false
end
res = Optimization.solve(prob, OptimizationOptimisers.Adam(); callback, maxiters=EPOCH, save_best=true)

t1 = 2197
Upinn, Vpinn, Tpinn = map(q -> permutedims(dropdims(q; dims=1)), res(Float32(t1),Float32.(X),Float32.(Y)))
# 표시 시점과 별개로, PDE를 학습한 구간 전체의 온도 오차를 평가한다.
lead = 0:Int(last(res[t])-t0)
rmse_ = [sqrt(mean(abs2, permutedims(dropdims(res(Float32(t0+d),Float32.(X),Float32.(Y); dv=T(t,x,y)); dims=1)) - _T[:,:,t0+d])) for d in lead]
plot(
    heatmap(X,Y,_T[:,:,t1],title="True T, t=$t1",clims=extrema(_T[:,:,t1])),
    heatmap(X,Y,Tpinn,title="PINN T, +$(t1-t0) days",clims=extrema(_T[:,:,t1])),
    plot(eachindex(L_), L_, scale=:log10, ticks = exp10.(-5:5), title="loss"),
    plot(lead,rmse_,title="T RMSE",xlabel="day lead",ylabel="RMSE (°C)",marker=:circle),
layout=(:,1), size=(600,600), legend=false)
