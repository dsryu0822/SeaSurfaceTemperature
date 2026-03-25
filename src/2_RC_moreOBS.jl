include("0_utils.jl")
include("1_datacall.jl")
function balloon(msg = "Done!")
    run(`powershell -command "[reflection.assembly]::loadwithpartialname('System.Windows.Forms') | Out-Null; \$n=new-object system.windows.forms.notifyicon; \$n.icon=[System.Drawing.SystemIcons]::Error; \$n.visible=\$true; \$n.showballoontip(5000,'알림','$msg',[system.windows.forms.tooltipicon]::Error); Start-Sleep -Seconds 5; \$n.dispose()"`)
    return nothing
end

# [T data][T .≥ Date(2022, 1, 1), :]
# _data = data[T .≥ Date(2019, 1, 1), :]

# id_observer = [1, 10, 141, 150]
# id_observer = [75]
# id_observer = id_flip([75])
# id_observer = id_flip([1])
# id_observer = unique([1:10:150; 1:10; 141:150; 10:10:150])
# id_observer = 10:10:150
id_observer = [1, 5, 10, 71, 80, 141, 145, 150]
begin
heatmap(whereis(id_observer), legend = :none, xticks = 1:15, yticks = 1:10, yflip = true)
savefig("C:/Users/rmsms/downloads/temp.svg")

resultRC_ = [DataFrame() for _ in 1:365]
@showprogress @threads for t0 = (1:365) # = rand(1:365)
    tspan = (t0:findfirst(x -> x == (T[t0] + Year(4) - Day(1)), T)) # [T _data][tspan, :]

    _U = Matrix(_data[tspan, id_observer])'
    _S = Matrix(_data[tspan, Not(id_observer)])'
    newU = Matrix(_data[last(tspan) .+ (1:30), id_observer])'
    actS = Matrix(_data[last(tspan) .+ (1:30), Not(id_observer)])'

    resultRC = DataFrame(Model = [], t0 = [], input = [], rmse1 = [], rmse7 = [], rmse30 = [], Hyperparameter = [])
    for _ in 1:1
        push!(resultRC, ["", 0, 0, 0, 0, 0, []])
    end
    for k in 1:nrow(resultRC)
        N = 1000; α = 1; ρ = 1
        # β = (120 < t0 ≤ 270) ? 1e-6 : 1e+1
        β = 1

        _rc = reservoir_computing(_U, _S; N, α, β, ρ, D)
        newS = _rc(newU)
        rmse_ = vec(rmse(newS, actS, dims = 1)); plot(rmse_)
        rmse1, rmse7, rmse30 = round.([rmse_[1], rmse_[7], rmse_[30]], digits = 3)

        resultRC[k, :] = ["RC", t0, length(tspan), rmse1, rmse7, rmse30, id_observer]
    end
    resultRC_[t0] = resultRC
    CSV.write("G:/seasurface/260326/resultRC$(string(hash(id_observer))).csv", [resultRC_...;], bom = true)
end

_resultRC = [resultRC_...;]
scatter(_resultRC.t0, _resultRC.rmse30, ylims = [0, 1.5], color = :black, xticks = 90*(0:4), legend = :none, yticks = [0:0.1:1; 1.5], title = "rmse_30: $(round(mean(_resultRC.rmse30), digits = 3))")
savefig("C:/Users/rmsms/downloads/temp.svg")
balloon()
end