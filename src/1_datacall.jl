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

@time "✅ nino3.4 jld2" JLD2.@load "G:/seasurface/nino3.4/data.jld2"

