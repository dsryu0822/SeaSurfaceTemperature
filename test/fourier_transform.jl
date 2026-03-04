using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :NoiseRobustDifferentiation,
    :Graphs, :Random, :LinearAlgebra] .|> string
try
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded")
catch e
    required = setdiff(packages, keys(Pkg.installed()))
    if !isempty(required) Pkg.add(required) end
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded after installation")
end

tensor2dataframe(tnsr) = DataFrame(stack(vec(eachslice(tnsr, dims = (1,2)))), :auto)
mae(x, y; dims = 2) = mean(abs, x - y; dims)
rmse(x, y; dims = 2) = sqrt.(mean(abs2, x - y; dims))
# encode(x) = *(Char.(x)...)
# decode(y) = Int64.(codepoint.(collect(y)))
DataFrame([rand(10) for _ in 1:5], :auto)
function add_diff(D::AbstractDataFrame; method = :FDM)
    dnames = "d" .* names(D)
    if method == :FDM
        return [DataFrame(diff(Matrix(D), dims = 1), dnames) D[1:(end-1), :]]
    else method == :TVD
        return [DataFrame([tvdiff(z, 10, 100, dx = 1) for z in eachcol(D)], dnames) D]
    end
end
function frequency(inventory)
    freq = Dict{eltype(inventory), Int64}()
    for item in inventory
        freq[item] = get(freq, item, 0) + 1
    end
    return freq
end

@load "data/data_tnsr.jld2"
T = CSV.read("data/data_GLBy0.08_expt_93.0.csv", DataFrame)[:, 1]
I, X = eachcol(CSV.read("data/lon.csv", DataFrame))
J, Y = eachcol(CSV.read("data/lat.csv", DataFrame))

μ = mean(tnsr); σ = std(tnsr)
data = tensor2dataframe(tnsr)
zk = ["z$k" for k in 1:Q]
rename!(data, zk)



plot(data.z1)

plot(real(ifft(filter(x -> real(x) > 100, fft(data.z1)))))


θ = 95
fftz = fft(data.z1)
tfftz = (percentile(abs.(fftz), θ) .< abs.(fftz)) .* fftz
abs.(ifft(tfftz))
plot(data.z1)
plot!(abs.(ifft(tfftz)))
struct prtc2
end


fften = fft(data.z1)

L = length(fften)
t_ = 1:L
a_ = 2real(fften[2:div(L,2)]) / L
b_ = -2imag(fften[2:div(L,2)]) / L

y = fill(real(fften[1]) / L, round(Int64, L))
for k in eachindex(a_)
    _sincos = sincospi.((2k/L)*t_)
    _sin = first.(_sincos)
    _cos = last.(_sincos)
    y .+= (a_[k] .* _cos) + (b_[k] .* _sin)
end
plot(data.z1)
plot!(y)


N = length(data.z1)
fftz = fft(data.z1)

futureN = 600  # 예측할 길이
pred = zeros(Float64, futureN)

for k in 1:N
    amp = abs(fftz[k]) / N
    phase = angle(fftz[k])
    freq = 2π*(k-1)/N
    pred .+= amp .* cos.(freq .* (0:futureN-1) .+ phase)
end
angle(fftz[3])
plot(pred)
plot!(data.z1[1:600])