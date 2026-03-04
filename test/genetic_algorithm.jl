# include("SINDy.jl")

using ColorSchemes


"""'''''''''''''''''''''''''''''''''''''''''''''

                genetic algorithm              

'''''''''''''''''''''''''''''''''''''''''''''"""
ranking(x; rev = true) = sortperm(sortperm(x; rev))

npop = 16; hpop = npop ÷ 2
lchsm = 32
lcnfg = nrow(cnfg)
epch = 50
crossover(m, f) = shuffle(unique([m; f]))[1:lchsm]
mutation(x) = [shuffle(x)[2:end]; rand(setdiff(1:lcnfg, x))]

resultGA = DataFrame(Model = [], t0 = [], input = [], R2 = [], rmse1 = [], rmse7 = [], rmse30 = [], Hyperparameter = [])
task1 = @async begin
    @showprogress @threads for t0 = _T
    t1 = t0 + input - 1

    chsm_ = sort.([shuffle(1:lcnfg)[1:lchsm] for _ in 1:npop])
    bestR2 = [.0]
    best = rand(chsm_)
    for _ in 1:epch
        fitness = [SINDy(ddata[t0:t1, :], vrbl, cnfg[chsm, :]).r2 for chsm in chsm_]
        if maximum(fitness) > bestR2[end]
            push!(bestR2, maximum(fitness))
            best = chsm_[argmax(fitness)]
        end

        rival = eachcol(reshape(shuffle(1:npop), 2, :))
        winners = []
        for (i, j) in rival
            push!(winners, fitness[i] > fitness[j] ? i : j)
        end
        lover = eachcol(reshape(shuffle(winners), 2, :))
        offspring = []
        for (i, j) in lover
            for _ in 1:4
                push!(offspring, crossover(chsm_[i], chsm_[j]) |> mutation)
            end
        end
        chsm_ = deepcopy(offspring)
    end

    selected = cnfg[best, :]
    f = SINDy(ddata[t0:t1, :], vrbl, selected)

    v = collect(ddata[t1, last(vrbl)])
    prdt = solve(f, v, 0:h:output)[1:Int64(1/h):end, :]
    test = Matrix(data[t1:(t1+output), :])

    pfmc = rmse(prdt, test; dims = 2)[2:end]
    push!(resultGA, ["SINDy", t0, input, maximum(bestR2), pfmc[1], pfmc[7], pfmc[30], selected.index])
end
CSV.write("dashboard/SINDy_genetic_algorithm32.csv", resultGA, bom = true)
end
resultGA
istaskstarted(task1)
istaskdone(task1)
istaskfailed(task1)

resultGA = CSV.read("dashboard/SINDy_genetic_algorithm32.csv", DataFrame)
t0mod = mod.(resultGA.t0, 365) / 365
scatter(1 .- resultGA.R2, resultGA.rmse1; xlabel = "1-R2", ylabel = "RMSE(1day)", title = "Genetic Algorithm Search Results", scale = :log10, msw = 0, mc = get.(Ref(clr_season), t0mod), ma = 0.5, legend = :none, size = [600, 600])
sort(resultGA, :R2, rev = true)
sort(resultGA, :R2, rev = true)
clr_season = cgrad([:dodgerblue, :green3, :orangered, :gold, :dodgerblue])


clr_season(0.1)