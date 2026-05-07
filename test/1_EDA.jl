include("0_utils.jl")
include("1_datacall.jl")

anime = @animate for t in eachindex(T)
    heatmap(X, Y, tnsr[:,:,t]', title="Time: $(T[t])", size = [480, 720], clim = extrema(tnsr))
end
gif(anime, "temp.gif", fps = 60)


plot(title="Temperature over time", xlabel="Date", ylabel="Temperature(°C)")
plot!(T, tnsr[1,8,:], alpha = 0.5, label = "(1,8)")
plot!(T, tnsr[5,8,:], alpha = 0.5, label = "(5,8)")
plot!(T, tnsr[10,8,:], alpha = 0.5, label = "(10,8)")

tnsr


@JLD2.load "G:/seasurface/seasurface_nino3.4/data_tnsr.jld2"