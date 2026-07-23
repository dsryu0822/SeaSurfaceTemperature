using JLD2, DataFrames, CSV, Plots

# @time nino34 = CSV.read("G:/seasurface/nino3.4/data.csv", DataFrame)
# X = [-5:0.04:5...;]
# Y = [-170:0.08:-120...;]
# T = nino34.t
# nino34 = nino34[:, 2:end]
# rename!(nino34, Symbol.(string.("z", 1:size(nino34, 2))))
# id_missing = (findall(eltype.(eachcol(nino34)) .!= Float64))
# nino34[:, id_missing] .= missing
# JLD2.@save "G:/seasurface/nino3.4/data.jld2" X Y T nino34 id_missing
mm = Plots.mm
@time jld2file = try
    JLD2.load("G:/seasurface/nino3.4/data.jld2")
catch e
    JLD2.load("/home/chaos/g/seasurface/nino3.4/data.jld2")
end
X = jld2file["X"]
Y = jld2file["Y"]
D = jld2file["D"]
nino34 = jld2file["nino34"]
id_missing = jld2file["id_missing"]
# @time "✅ nino3.4 jld2" JLD2.@load "G:/seasurface/nino3.4/datam.jld2"

# default(framestyle = :box, legend = :none)

# """''''''''''''''''''''''''''''''''''''''''''''''''''

#                         EDA

# ''''''''''''''''''''''''''''''''''''''''''''''''''"""
# μ = mean(Matrix(nino34[:, id_target]), dims = 2)
# σ = std(Matrix(nino34[:, id_target]), dims = 2)
# plot(legend = :none, size = [600, 200], xlims = [2019, 2026.4])
# plot!(2019 .+ ((1:length(μ))/365), μ, xticks = 2019:2027, lw = 2, color = :black, alpha = 0.5)
# plot!(2019 .+ ((1:length(μ))/365), μ - σ, fillrange = μ .+ σ, fillcolor = :black, fillalpha = 0.2, label = :none, alpha = 0)
# vspan!([2019, 2025], color = 1, alpha = 0.2)
# vspan!([2025, 2027], color = 2, alpha = 0.2)
