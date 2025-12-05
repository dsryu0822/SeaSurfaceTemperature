using Base.Threads: @threads, nthreads # Base.Threads.nthreads()
import Pkg
packages = [:CSV, :DataFrames, :Dates, :ProgressMeter, :FFTW, :StatsBase,
    :LinearAlgebra, :Distances, :JLD2, :UnicodePlots] .|> string
try
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded")
catch e
    required = setdiff(packages, keys(Pkg.installed()))
    if !isempty(required) Pkg.add(required) end
    @time eval(Meta.parse("using $(join(packages, ", "))"))
    println("All packages loaded after installation")
end
include("C:/Users/rmsms/OneDrive/lab/DataDrivenModel/core/header.jl")

tensor2dataframe(tnsr) = DataFrame(stack(vec(eachslice(tnsr, dims = (1,2)))), :auto)
add_diff(D::AbstractDataFrame) = [DataFrame(diff(Matrix(D), dims = 1), "d" .* names(D)) D[1:(end-1), :]]
mae(x, y; dims = 2) = mean(abs, x - y; dims)
rmse(x, y; dims = 2) = sqrt.(mean(abs2, x - y; dims))
encode(x) = *(Char.(x)...)
decode(y) = Int64.(codepoint.(collect(y)))


@load "data/data_tnsr.jld2"
T = CSV.read("data/data_GLBy0.08_expt_93.0.csv", DataFrame)[:, 1]
I, X = eachcol(CSV.read("data/lon.csv", DataFrame))
J, Y = eachcol(CSV.read("data/lat.csv", DataFrame))

data = tensor2dataframe(tnsr)
dim = ncol(data)
zk = ["z$k" for k in 1:dim]
rename!(data, zk)
ddata = add_diff(data)

vrbl = (names(ddata)[1:dim], names(ddata)[(dim+1):end])
cnfg = cook(zk; poly = 0:2)

input = 30
output = 30
h = 1e-2

m = nrow(cnfg)
function mutate(x)
    y = deepcopy(x)
    if rand() < (1/(length(x)))
        push!(y, rand(setdiff(1:m, y)))
    else
        shuffle!(y)
        pop!(y)
    end
    return sort(y)
end
mix(x,y) = unique(shuffle([x; y]))[1:(end-1)]
function bisect(df::DataFrame)
    _df = shuffle(df)
    half = nrow(_df) ÷ 2
    top = _df[1:half, :]
    bottom = _df[(half+1):end, :]
    return top, bottom
end

n = 32
half = n ÷ 2

chromosome = [mutate([]) for _ in 1:n]
best = DataFrame(Hyperparameter = [], t0 = [], rmse1 = [], rmse7 = [], rmse30 = [])
for gen in 1:10
    arena = DataFrame(Hyperparameter = [], t0 = [], rmse1 = [], rmse7 = [], rmse30 = [])
    for gene in chromosome
        try
            selected = sort(cnfg[gene, :], :index)
            t0 = rand(eachindex(T))
            t1 = t0 + input - 1
            f = SINDy(ddata[t0:t1, :], vrbl, selected)
            v = collect(ddata[t1, last(vrbl)])

            prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
            test = Matrix(data[t1:(t1+output), :])

            pfmc = rmse(prdt, test; dims = 2)[2:end]
            push!(arena, [encode(selected.index), t0, pfmc[1], pfmc[7], pfmc[30]])
        catch e
            push!(arena, [encode(selected.index), t0, NaN, NaN, NaN])
        end
    end
    replace!(arena.rmse1, NaN => Inf)
    replace!(arena.rmse7, NaN => Inf)
    replace!(arena.rmse30, NaN => Inf)
    push!(best, sort(arena, :rmse1)[1, :])
    push!(best, sort(arena, :rmse7)[1, :])
    push!(best, sort(arena, :rmse30)[1, :])
    println("""──────────────────────────
    Generation $gen completed.
    Best RMSE1: $(round(best.rmse1[end], digits=4))
    Best RMSE7: $(round(best.rmse7[end], digits=4))
    Best RMSE30: $(round(best.rmse30[end], digits=4))""")
    # lineplot(1:gen, [best.rmse30[3:3:end] best.rmse7[2:3:end] best.rmse1[1:3:end]], color=[:green :yellow :red], name = ["rmse30" "rmse7" "rmse1"])

    wedding = sort(arena, :rmse30)[(half+1):end, :]
    # black, white = bisect(arena)
    # fight = vec(sum(Matrix(black[:, (end-2):end]) .< Matrix(white[:, (end-2):end]), dims = 2) .< 1.5)
    # wedding = [black[fight, :]; white[.!fight, :]]

    male, female = bisect(wedding)
    offspring1 = mutate.(mix.(decode.(male.Hyperparameter), decode.(female.Hyperparameter)))
    offspring2 = mutate.(mix.(decode.(male.Hyperparameter), decode.(female.Hyperparameter)))
    chromosome = [decode.(wedding.Hyperparameter); offspring1; offspring2]
end
plot(length.(decode.(best.Hyperparameter)))

plot(best.rmse1[1:3:end])
plot(best.rmse30[3:3:end])