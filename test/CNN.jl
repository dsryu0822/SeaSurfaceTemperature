using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :Flux, :CUDA] .|> string
try
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded")
catch e
    required = setdiff(packages, keys(Pkg.installed()))
    if !isempty(required) Pkg.add(required) end
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded after installation")
end

mae(x, y; dims = 2) = mean(abs, x - y; dims)
rmse(x, y; dims = 2) = sqrt.(mean(abs2, x - y; dims))

@assert CUDA.functional(true)

@load "data/data_tnsr.jld2"
X = cu(Float32.(reshape(tnsr[:,:,1:(end-1)], 10, 15, 1, :)))
Y = cu(Float32.(reshape(tnsr[:,:,2:end], 10, 15, 1, :)))
# DataLoader constructed from GPU arrays (no need to pipe the loader to gpu)
loader = Flux.DataLoader((X, Y), batchsize=64, shuffle=true);
Loss(f, x, y) = Flux.mse(f(x), y)

CNN = Chain(
    Conv((2,2),   1 => 100, relu; pad = SamePad()),
    Conv((2,2), 100 => 100, relu; pad = SamePad()),
    Conv((2,2), 100 => 100, relu; pad = SamePad()),
    Conv((2,2), 100 =>   1; pad = SamePad()),
) |> gpu # CNN(CUDA.rand(Float32, 10, 15, 1, 1))
optimizer = Flux.setup(Flux.Adam(0.01), CNN); # will store optimiser momentum, etc.

losses = [Inf]
topCNN = deepcopy(CNN)
for epoch in 1:100
    Flux.train!(Loss, CNN, loader, optimizer)
    loss = Loss(CNN, X, Y)
    println("Epoch $epoch: Loss = $loss")
    if loss < minimum(losses)
        topCNN = deepcopy(CNN)
    end
    push!(losses, loss)  # logging, outside gradient context
end
cnn(x) = reshape(Array(topCNN(reshape(cu(x), 10, 15, 1, 1))), 10, 15)

heatmap(topCNN(cu(reshape(X[:,:,:,1], 10, 15, 1, 1)))[:,:,1,1] |> cpu)
y1 = topCNN(cu(reshape(X[:,:,:,1], 10, 15, 1, 1)))[:,:,1,1] |> cpu
plot(
    heatmap(y1),
    heatmap(Y[:,:,1,2] |> cpu),
)
Base.Flatten
rmse(vec(y1)', vec(Y[:,:,:,2] |> cpu)')









@load "data/data_tnsr.jld2"
input = 30

X = cu(Float32.(stack([reshape(tnsr[:,:,t:(t+input-1)], 10, 15, input) for t in 1:(size(tnsr,3)-input)])))
Y = cu(Float32.(reshape(tnsr[:,:,(input+1):end], 10, 15, 1, :)))
# DataLoader constructed from GPU arrays (no need to pipe the loader to gpu)
loader = Flux.DataLoader((X, Y), batchsize=64, shuffle=true);
Loss(f, x, y) = Flux.mse(f(x), y)

CNN = Chain(
    Conv((2,2), input => 100, relu; pad = SamePad()),
    Conv((2,2),   100 => 100, relu; pad = SamePad()),
    Conv((2,2),   100 => 100, relu; pad = SamePad()),
    Conv((2,2),   100 =>   1; pad = SamePad()),
) |> gpu # CNN(CUDA.rand(Float32, 10, 15, 1, 1))
optimizer = Flux.setup(Flux.Adam(0.001), CNN); # will store optimiser momentum, etc.

losses = [Inf]
topCNN = deepcopy(CNN)
@async for epoch in 1:90_000
    Flux.train!(Loss, CNN, loader, optimizer)
    loss = Loss(CNN, X, Y)
    println("Epoch $epoch: Loss = $loss")
    if loss < minimum(losses)
        topCNN = deepcopy(CNN)
    end
    push!(losses, loss)  # logging, outside gradient context
end
cnn(x) = reshape(topCNN(reshape(x, [size(x)...; 1]...)), size(x)[1:2]...) |> cpu

plot(losses, scale = :log10)
rsme_ = []
@showprogress for t0 = 1:2000
    pred = cnn(X[:,:,:,t0])
    actl = Y[:,:,1,t0] |> cpu
    push!(rsme_, only(rmse(vec(pred)', vec(actl)')))
end
Loss(topCNN, X, Y)


t0 = 1000
# sqrt(sum(abs2.(topCNN(X) - Y)) / prod(size(Y)))

pred = reshape(topCNN(X)[:,:,:,t0], 10, 15) |> cpu
actl = Y[:,:,1,t0] |> cpu
plot(
    heatmap(pred),
    heatmap(actl),
    clims = extrema([pred; actl])
)
rmse(vec(pred)', vec(actl)')
minimum(losses)

# @save "arc/30days.16.jld2" topCNN cnn losses
plot(losses)

mean(rsme_)
scatter(rsme_, yscale = :log10)

sqrt(.101)