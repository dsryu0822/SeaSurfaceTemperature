include("0_utils.jl")
include("1_datacall.jl")
function balloon(msg = "Done!")
    run(`powershell -command "[reflection.assembly]::loadwithpartialname('System.Windows.Forms') | Out-Null; \$n=new-object system.windows.forms.notifyicon; \$n.icon=[System.Drawing.SystemIcons]::Error; \$n.visible=\$true; \$n.showballoontip(5000,'알림','$msg',[system.windows.forms.tooltipicon]::Error); Start-Sleep -Seconds 5; \$n.dispose()"`)
    return nothing
end

using PRIMA

# [T data][T .≥ Date(2022, 1, 1), :]
_data = data[T .≥ Date(2019, 1, 1), :]

id_observer = [1, 10, 141, 150]
id_observer = [1, 5, 10, 71, 80, 141, 145, 150]

t0 = 1
tspan = (t0:findfirst(x -> x == (T[t0] + Year(3) - Day(1)), T)) # [T _data][tspan, :]_U = Matrix(_data[tspan, id_observer])'
_U3 = Matrix(_data[tspan, id_observer])'
_S3 = Matrix(_data[tspan, Not(id_observer)])'
newU3 = Matrix(_data[last(tspan) .+ (1:365), id_observer])'
actS3 = Matrix(_data[last(tspan) .+ (1:365), Not(id_observer)])'

ReLu(x) = max(0, x)
function reservoiral(NDαβρσξ)
    N, D, α, β, ρ, σ, ξ = exp10.(NDαβρσξ)
    N = round(Int64, N)
    α = atan(log10(α)) + 1

    _rc = reservoir_computing(_U3, _S3; N, D, α, β, ρ, σ, ξ)
    return mean(rmse(_rc(newU3), actS3, dims = 1))
end

# NDαβρσξ_0 = [50, 2, 1, -6, 0, 1, 0]
# @time NDαβρσξ_opt, info = prima(reservoiral, NDαβρσξ_0,
#     xl = [  0,   0, 0, -Inf, -Inf,   0, -Inf],
#     xu = [Inf, Inf, 1,  Inf,  Inf, Inf,  Inf])
NDαβρσξ_0 = [2, 0, 0, -2, 0, 0, 0]
@time NDαβρσξ_opt, info = prima(reservoiral, NDαβρσξ_0)
reservoiral(NDαβρσξ_opt)
_NDαβρσξ_opt = (; N = round(Int64, exp10(NDαβρσξ_opt[1])), D = exp10(NDαβρσξ_opt[2]), α = atan(log10(exp10(NDαβρσξ_opt[3]))) + 1, β = exp10(NDαβρσξ_opt[4]), ρ = exp10(NDαβρσξ_opt[5]), σ = exp10(NDαβρσξ_opt[6]), ξ = exp10(NDαβρσξ_opt[7]))

tspan = (t0:findfirst(x -> x == (T[t0] + Year(4) - Day(1)), T)) # [T _data][tspan, :]_U = Matrix(_data[tspan, id_observer])'
_U = Matrix(_data[tspan, id_observer])'
_S = Matrix(_data[tspan, Not(id_observer)])'
newU = Matrix(_data[last(tspan) .+ (1:365), id_observer])'
actS = Matrix(_data[last(tspan) .+ (1:365), Not(id_observer)])'

rc = reservoir_computing(_U, _S)
rmse_ = vec(rmse(rc(newU), actS, dims = 1)); # plot(rmse_)
mean(rmse_)

rc = reservoir_computing(_U, _S; _NDαβρσξ_opt...)
rmse_ = vec(rmse(rc(newU), actS, dims = 1)); # plot(rmse_)
mean(rmse_)
rc |> print

