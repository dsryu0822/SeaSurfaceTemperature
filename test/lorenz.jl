# include("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core/header.jl")
# CSV.write("lorenz.csv", factory_lorenz(DataFrame, 28; ic = [1,1,1], tspan = [0, 100], dt = 0.02))
# data = CSV.read("lorenz.csv", DataFrame)
standardize(x) = (x .- mean(x)) ./ std(x)
for col in eachcol(data)
    col .= standardize(col)
end
_U = [data.x data.y]'
_S = data.z'
@time rc_ = reservoir_computing(_U[:, 1:4000], _S[:, 1:4000], N = 200, D = 10, α = 1)
plot(vec(rc_(_U[:, 4001:end])))
plot!(vec(_S[:, 4001:end]))

